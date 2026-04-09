from flask import Flask, request, jsonify
import yfinance as yf

# ייבוא הפונקציות המעודכנות מהמנוע הראשי
from main import run_rnn_analysis, run_trend_analysis, run_nlp_analysis, apply_trading_rules

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # 1. שולפים את שם המניה שה-Java שלח (למשל ?symbol=TSLA)
    symbol = request.args.get('symbol')

    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400

    symbol = symbol.upper()
    print("=" * 50)
    print(f" מנתח את מניית: {symbol} (בקשה התקבלה מה-Java!)")
    print("=" * 50)

    try:
        # 2. מורידים נתונים טריים למניה המבוקשת
        df = yf.Ticker(symbol).history(period="5y")
        if df.empty:
            return jsonify({"error": f"No data found for {symbol}"}), 404

        # 3. מריצים את שלושת מודלי ה-AI
        # שים לב: כאן אנחנו קולטים שני ערכים מה-RNN כפי שעדכנו ב-MAIN
        rnn_prob, predicted_price = run_rnn_analysis(symbol, df)
        trend_prob = run_trend_analysis(symbol, df)
        nlp_score = run_nlp_analysis(symbol)

        # 4. מעבירים למנוע החוקים לקבלת החלטה סופית
        decision, final_prob, reason = apply_trading_rules(rnn_prob, trend_prob, nlp_score)

        current_price = df['Close'].iloc[-1]

        # 5. אורזים את הכל כ-JSON ושולחים חזרה ל-Java
        # הוספנו את predictedPrice ו-expectedChangePercent
        result = {
            "symbol": symbol,
            "currentPrice": round(current_price, 2),
            "predictedPrice": round(predicted_price, 2),
            "expectedChangePercent": round(((predicted_price - current_price) / current_price) * 100, 2),
            "systemScore": round(final_prob * 100, 1),
            "recommendation": decision,
            "reason": reason
        }

        return jsonify(result)

    except Exception as e:
        print(f"שגיאה בזמן העיבוד: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # מפעילים את שרת ה-Flask על פורט 5000
    app.run(port=5000, debug=True)