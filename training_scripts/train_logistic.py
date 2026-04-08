import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# שלב 1: איסוף נתונים
# ---------------------------------------------------------
ticker_symbol = "AAPL"
print(f"מושך נתונים עבור {ticker_symbol}...")
ticker = yf.Ticker(ticker_symbol)
df = ticker.history(period="5y")

# ---------------------------------------------------------
# שלב 2: הנדסת מאפיינים (הגרסה הקלאסית והטובה)
# ---------------------------------------------------------
print("מחשב אינדיקטורים טכניים...")

# RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# SMA & Ratio
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['Price_vs_SMA'] = df['Close'] / df['SMA_50']

# ---------------------------------------------------------
# שלב 3: Target
# ---------------------------------------------------------
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
df.dropna(inplace=True)

# ---------------------------------------------------------
# שלב 4: חלוקה (בלי נרמול - הוא רק הפריע כאן)
# ---------------------------------------------------------
feature_cols = ['RSI', 'MACD', 'Signal_Line', 'Price_vs_SMA']
X = df[feature_cols]
y = df['Target']

# חלוקה כרונולוגית
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ---------------------------------------------------------
# שלב 5: אימון המודל (עם משקלים מאוזנים!)
# ---------------------------------------------------------
print("מאמן את המודל הלוגיסטי...")
# זה הסוד להצלחה - class_weight='balanced'
model = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# שלב 6: תוצאות
# ---------------------------------------------------------
y_pred = model.predict(X_test)

print(f"Final Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

# ---------------------------------------------------------
# שלב 7: שמירת המודל (התוספת שלנו)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
os.makedirs(MODELS_DIR, exist_ok=True)

model_path = os.path.join(MODELS_DIR, 'logistic_model.pkl')
joblib.dump(model, model_path)
print(f"\n הצלחה! המודל הלוגיסטי נשמר בכתובת: {model_path}")

# הצגת המטריצה
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title('Final Trend Prediction (Balanced)')
plt.show()