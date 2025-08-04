from flask import Flask, request, jsonify
import speech_recognition as sr
import tempfile

app = Flask(__name__)

@app.route("/")
def index():
    return "✅ Kannada STT API is running!"

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]

    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        audio_file.save(temp.name)

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp.name) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data, language="kn-IN")
            return jsonify({"transcript": text})
        except sr.UnknownValueError:
            return jsonify({"error": "Speech not understood"}), 400
        except sr.RequestError as e:
            return jsonify({"error": f"Speech recognition error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
