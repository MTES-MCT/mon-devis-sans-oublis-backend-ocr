from flask import Flask, request, jsonify, abort
import os

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")

@app.before_request
def check_api_key():
    if request.path == "/ocr":
        api_key = request.headers.get("X-API-Key")
        if api_key != API_KEY:
            return jsonify(
                status="error",
                code=401,
                error="Unauthorized",
                message="API key missing or invalid"
            ), 401

@app.route("/ocr", methods=["POST"])
def ocr():
    data = request.get_json()
    text = data.get("text", "")
    return jsonify(status="success", text=text)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=os.getenv("PORT", 80),
        debug=os.getenv("FLASK_ENV") != "production"
    )
