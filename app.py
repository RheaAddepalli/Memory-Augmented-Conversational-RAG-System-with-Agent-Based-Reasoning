









from flask import Flask, request, jsonify
import subprocess
import os
import json
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return "DocMind Flask Server Running 🚀"


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # Run your existing pipeline
        result = subprocess.run(
            [os.path.join("venv_old", "Scripts", "python.exe"), "process_pdf.py", filepath],
            capture_output=True,
            text=True
        )
        print("\n===== RAW PYTHON OUTPUT =====\n")
        print(result.stdout)
        print("\n===== ERRORS =====\n")
        print(result.stderr)


        output = result.stdout.strip()

        # Extract JSON from output
        match = re.search(r"\{.*\}", output, re.S)

        if match:
            data = json.loads(match.group())
            return jsonify(data)
        else:
            return jsonify({"status": "error", "raw": output})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
