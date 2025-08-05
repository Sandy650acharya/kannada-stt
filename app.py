from flask import Flask, request, jsonify
import speech_recognition as sr
import tempfile
from pydub import AudioSegment, silence
import os

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
        wav_path = temp.name

    try:
        # Load and preprocess audio with pydub
        audio = AudioSegment.from_wav(wav_path)

        # Trim silence longer than 10s (10000ms)
        chunks = silence.split_on_silence(
            audio,
            min_silence_len=10000,  # 10 seconds of silence
            silence_thresh=audio.dBFS - 14,
            keep_silence=500  # keep slight context
        )

        if not chunks:
            return jsonify({"error": "Audio contains too much silence or is unclear."}), 400

        # Combine chunks back into one short audio (stop at silence)
        processed_audio = chunks[0]  # only first speech part before long silence
        processed_path = wav_path.replace(".wav", "_processed.wav")
        processed_audio.export(processed_path, format="wav")

        # Recognize speech
        recognizer = sr.Recognizer()
        with sr.AudioFile(processed_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data, language="kn-IN")
        return jsonify({"transcript": text})

    except sr.UnknownValueError:
        return jsonify({"error": "Speech not understood"}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition error: {str(e)}"}), 500
    except Exception as ex:
        return jsonify({"error": f"Internal server error: {str(ex)}"}), 500
    finally:
        # Clean up temp files
        if os.path.exists(wav_path):
            os.remove(wav_path)
        if 'processed_path' in locals() and os.path.exists(processed_path):
            os.remove(processed_path)

if __name__ == "__main__":
    app.run(debug=True)
