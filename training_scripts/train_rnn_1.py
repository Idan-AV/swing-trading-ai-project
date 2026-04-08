import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- הוספת נתיבי שמירה ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
os.makedirs(MODELS_DIR, exist_ok=True)

# שלב 1: איסוף נתונים
ticker_symbol = "AAPL"
print(f"מושך נתונים עבור {ticker_symbol}...")
ticker = yf.Ticker(ticker_symbol)
df = ticker.history(period="5y")

# שלב 2: עיבוד מקדים
df['Volume_Log'] = np.log1p(df['Volume'])
features = ['Open', 'High', 'Low', 'Close', 'Volume_Log']
target_col = 'Close'

# שלב 3: הכנת הנתונים (Scaling & Sequencing)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[features])

def create_dataset(dataset, look_back=60):
    X, y = [], []
    close_idx = features.index('Close')
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, :])
        y.append(dataset[i, close_idx])
    return np.array(X), np.array(y)

LOOK_BACK = 60
X, y = create_dataset(scaled_data, LOOK_BACK)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# שלב 4: בניית מודל ה-RNN
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# שלב 5: אימון
print("מתחיל אימון... זה ייקח דקה או שתיים")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# --- שלב השמירה (זה מה שהוספתי) ---
model_path = os.path.join(MODELS_DIR, 'rnn_model.h5')
model.save(model_path)
print(f"\n הצלחה! המודל נשמר בכתובת: {model_path}")

# שלב 6: הצגת תוצאות (גרף)
predictions = model.predict(X_test)
dummy_array = np.zeros((len(predictions), len(features)))
dummy_array[:, features.index('Close')] = predictions.flatten()
inverse_predictions = scaler.inverse_transform(dummy_array)[:, features.index('Close')]

dummy_array_y = np.zeros((len(y_test), len(features)))
dummy_array_y[:, features.index('Close')] = y_test.flatten()
real_prices = scaler.inverse_transform(dummy_array_y)[:, features.index('Close')]

plt.figure(figsize=(12, 6))
plt.plot(real_prices, color='black', label='Real Price')
plt.plot(inverse_predictions, color='green', label='RNN Prediction')
plt.title(f'Machine 1: {ticker_symbol} Prediction')
plt.legend()
plt.show()