import os
import pandas as pd
import numpy as np
import requests
import random
import joblib
import nltk
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# הגדרת נתיבי שמירה לתיקיית הפרויקט בכונן D
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(MODELS_DIR, exist_ok=True)

# הכרחת NLTK להוריד את המילונים לכונן D (ולא ל-C)
NLTK_DIR = os.path.join(DATA_DIR, 'nltk_data')
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

print("מוריד מילונים (NLTK) לכונן D...")
nltk.download('wordnet', download_dir=NLTK_DIR, quiet=True)
nltk.download('omw-1.4', download_dir=NLTK_DIR, quiet=True)
nltk.download('stopwords', download_dir=NLTK_DIR, quiet=True)
nltk.download('punkt', download_dir=NLTK_DIR, quiet=True)
stop_words = set(stopwords.words('english'))


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower():
                synonyms.add(candidate)
    return list(synonyms)


def augment_sentence(text):
    words = text.split()
    if len(words) < 3: return text
    new_words = words.copy()
    for _ in range(5):
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if word.lower() not in stop_words and word.isalpha():
            synonyms = get_synonyms(word)
            if synonyms:
                new_words[idx] = random.choice(synonyms)
                return " ".join(new_words)
    return text


def load_and_train_model():
    print("\nשלב 1: מאמן את המודל על נתוני ה-CSV...")

    # הנתיב המדויק לקובץ במחשב שלך
    filename = r"C:\Users\avulo\Downloads\all-data.csv"

    try:
        df = pd.read_csv(filename, encoding='ISO-8859-1', header=None, names=['label', 'text'])
    except FileNotFoundError:
        print(f"שגיאה: הקובץ לא נמצא בנתיב: {filename}")
        return None, None

    # איזון נתונים (Augmentation)
    counts = df['label'].value_counts()
    max_count = counts.max()
    new_rows = []
    for label in counts.index:
        diff = max_count - counts[label]
        if diff > 0:
            subset = df[df['label'] == label]
            for _ in range(diff):
                random_text = subset.sample(n=1, replace=True)['text'].iloc[0]
                new_rows.append({'label': label, 'text': augment_sentence(random_text)})

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # וקטוריזציה
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['target'] = df['label'].map(label_map)

    vectorizer = TfidfVectorizer(max_features=2500, stop_words='english')
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df['target']

    # אימון - הוסרה המילה multi_class שהקריסה את הקוד
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)

    # בדיקת דיוק
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"אימון הושלם! דיוק המודל: {acc:.2f}")

    # שמירת המודל
    model_path = os.path.join(MODELS_DIR, 'nlp_model.pkl')
    vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"המודל והמילון נשמרו בתיקיית: {MODELS_DIR}")

    return model, vectorizer


def get_live_sentiment(symbol, api_key, model, vectorizer):
    print(f"\nשלב 2: מתחבר ל-Finnhub למשיכת חדשות זמן-אמת עבור {symbol}...")
    today = datetime.now().strftime('%Y-%m-%d')
    last_week = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={last_week}&to={today}&token={api_key}"

    try:
        response = requests.get(url)
        news_data = response.json()
    except Exception as e:
        print(f"שגיאה בחיבור ל-Finnhub: {e}")
        return 1

    if not news_data:
        print("לא נמצאו חדשות חדשות. מחזיר ניטרלי.")
        return 1

    print(f"  -> נמצאו {len(news_data)} כותרות. מנתח את ה-5 העדכניות ביותר:")

    scores = []
    for item in news_data[:5]:
        text = f"{item['headline']}. {item['summary']}"
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        scores.append(prediction)

        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        dt = datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d')
        print(f"  [{dt}] {sentiment_map[prediction]}: {item['headline'][:60]}...")

    if scores:
        final_score = int(round(np.mean(scores)))
        final_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        print(f"\nסיכום סנטימנט משוקלל ל-{symbol}: {final_score} ({final_map[final_score]})")
        return final_score
    return 1


# הרצה ראשית
nlp_model, tfidf_vectorizer = load_and_train_model()

if nlp_model:
    MY_API_KEY = "d60drlhr01qto1rcoqp0d60drlhr01qto1rcoqpg"
    current_sentiment = get_live_sentiment("NVDA", MY_API_KEY, nlp_model, tfidf_vectorizer)