import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import joblib
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- הגדרות נתיבים ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
RNN_MODEL_FILE = os.path.join(MODELS_DIR, 'rnn_model.h5')
TREND_MODEL_FILE = os.path.join(MODELS_DIR, 'logistic_model.pkl')
NLP_MODEL_FILE = os.path.join(MODELS_DIR, 'nlp_model.pkl')
NLP_VECTORIZER_FILE = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')

API_KEY_FINNHUB = "d60drlhr01qto1rcoqp0d60drlhr01qto1rcoqpg"

print("\nמאתחל מערכת מסחר: [RNN Price] + [Trend Sigmoid] + [Custom NLP]...")

# ========================================================================
# שלב 1: טעינת המודלים (Load Models)
# ========================================================================
try:
    print("טוען מודל RNN (מכונה 1)...")
    rnn_model = load_model(RNN_MODEL_FILE)
    print("טוען מודל מגמה - Logistic Regression (מכונה 2)...")
    trend_model = joblib.load(TREND_MODEL_FILE)
    print("טוען מודל NLP מותאם אישית והמילון שלו (מכונה 3)...")
    nlp_model = joblib.load(NLP_MODEL_FILE)
    tfidf_vectorizer = joblib.load(NLP_VECTORIZER_FILE)
    print("כל המודלים נטענו בהצלחה!\n")
except Exception as e:
    print(f"שגיאה בטעינת המודלים: {e}")
    exit()


# ========================================================================
# שלב 2: פונקציות הניתוח (The Sub-Systems)
# ========================================================================

def run_rnn_analysis(ticker, df):
    df_rnn = df.copy()
    df_rnn['Volume_Log'] = np.log1p(df_rnn['Volume'])
    features = ['Open', 'High', 'Low', 'Close', 'Volume_Log']
    data = df_rnn[features].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    last_60_days = scaled_data[-60:]
    X_input = np.array([last_60_days])

    try:
        pred_scaled = rnn_model.predict(X_input, verbose=0)[0][0]
        dummy_array = np.zeros((1, len(features)))
        dummy_array[0, features.index('Close')] = pred_scaled
        pred_price = scaler.inverse_transform(dummy_array)[0, features.index('Close')]
    except:
        return 0.5  # Default in case of error

    current_price = df['Close'].iloc[-1]
    expected_change = (pred_price - current_price) / current_price

    # שימוש בפונקציית סיגמואיד להמרת אחוז השינוי להסתברות (בדיוק כמו בספר)
    k = 40
    rnn_prob = 1 / (1 + np.exp(-k * expected_change))

    print(f" [RNN Info] Current Price: ${current_price:.2f} | Predicted Price: ${pred_price:.2f}")
    return rnn_prob


def run_trend_analysis(ticker, df):
    df_trend = df.copy()
    delta = df_trend['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_trend['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df_trend['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_trend['Close'].ewm(span=26, adjust=False).mean()
    df_trend['MACD'] = exp1 - exp2
    df_trend['Signal_Line'] = df_trend['MACD'].ewm(span=9, adjust=False).mean()
    df_trend['SMA_50'] = df_trend['Close'].rolling(window=50).mean()
    df_trend['Price_vs_SMA'] = df_trend['Close'] / df_trend['SMA_50']

    df_trend.dropna(inplace=True)
    feature_cols = ['RSI', 'MACD', 'Signal_Line', 'Price_vs_SMA']
    last_row = df_trend[feature_cols].iloc[[-1]]

    try:
        trend_prob = trend_model.predict_proba(last_row)[0][1]
    except:
        trend_prob = 0.55

    return trend_prob


def run_nlp_analysis(ticker):
    today = datetime.now().strftime('%Y-%m-%d')
    last_week = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={last_week}&to={today}&token={API_KEY_FINNHUB}"

    try:
        res = requests.get(url).json()
    except:
        return 0

    if not res: return 0
    scores = []

    for item in res[:5]:
        text = f"{item['headline']}. {item['summary']}"
        try:
            vec = tfidf_vectorizer.transform([text])
            prediction = nlp_model.predict(vec)[0]
            val = 0
            if prediction == 2:
                val = 1
            elif prediction == 0:
                val = -1
            scores.append(val)
        except:
            continue

    if not scores: return 0
    avg = np.mean(scores)
    return 1 if avg > 0.2 else (-1 if avg < -0.2 else 0)


# ========================================================================
# שלב 3: מנוע החוקים הראשי (Rule-Based System) - כמו בספר!
# ========================================================================

def apply_trading_rules(rnn_prob, trend_prob, nlp_score):
    # שקלול מתמטי: 45% מחיר ו-55% מגמה (לפי עמוד 100 בספר)
    combined_prob = (rnn_prob * 0.45) + (trend_prob * 0.55)

    # 1. וטו של חדשות רעות
    if nlp_score == -1:
        return "AVOID", combined_prob, "חדשות שליליות - וטו הופעל."
    # 2. סינון שוק דובי
    if trend_prob < 0.45:
        return "HOLD", combined_prob, "המגמה הכללית שלילית. לא קונים."
    # 3. אישור משולש מושלם
    if combined_prob > 0.65 and nlp_score == 1:
        return "STRONG BUY", combined_prob, "טכני חזק + ציפייה לעליית מחיר + חדשות טובות."
    # 4. סיגנל טכני רגיל
    if combined_prob >= 0.60:
        return "BUY", combined_prob, "אישור ממודל ה-RNN (מחיר) ומודל הלוגיסטי (מגמה)."
    # 5. מכירה
    if combined_prob < 0.40:
        return "SELL", combined_prob, "מגמה יורדת ומודל ה-RNN חוזה ירידה."

    return "WAIT", combined_prob, "אין סיגנל מובהק. המתנה לכניסה."


def evaluate_stock(ticker):
    print(f"\n" + "=" * 50)
    print(f" מנתח את מניית: {ticker}")
    print("=" * 50)

    try:
        df = yf.Ticker(ticker).history(period="5y")
    except Exception as e:
        print(f"שגיאה במשיכת נתונים: {e}")
        return

    rnn_prob = run_rnn_analysis(ticker, df)
    trend_prob = run_trend_analysis(ticker, df)
    nlp_score = run_nlp_analysis(ticker)

    decision, final_prob, reason = apply_trading_rules(rnn_prob, trend_prob, nlp_score)

    print("-" * 50)
    print(f" ציון מערכת משוקלל: {final_prob:.1%}")
    print(f" החלטת המערכת: {decision}")
    print(f" נימוק (Rule Set): {reason}")
    print("=" * 50)


if __name__ == "__main__":
    while True:
        target_symbol = input("\nהכנס סימול מניה (או 'Q' ליציאה): ").strip().upper()
        if target_symbol == 'Q':
            break
        if target_symbol:
            evaluate_stock(target_symbol)