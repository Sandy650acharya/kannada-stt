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
    try:
        # Save incoming file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            audio_file.save(temp.name)
            wav_path = temp.name

        # Load audio with pydub and convert to mono
        audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(16000)

        # Split on silence - dynamic chunking
        chunks = silence.split_on_silence(
            audio,
            min_silence_len=1000,  # 1 sec silence as chunk separator
            silence_thresh=audio.dBFS - 14,
            keep_silence=300
        )

        if not chunks:
            return jsonify({"error": "No speech detected. Audio may be silent or unclear."}), 400

        recognizer = sr.Recognizer()
        full_transcription = []

        for i, chunk in enumerate(chunks):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
                chunk.export(chunk_file.name, format="wav")
                try:
                    with sr.AudioFile(chunk_file.name) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data, language="kn-IN")
                        full_transcription.append(text)
                except sr.UnknownValueError:
                    full_transcription.append("[Unrecognized]")
                except sr.RequestError as e:
                    return jsonify({"error": f"Google API error: {str(e)}"}), 500
                finally:
                    os.remove(chunk_file.name)

        final_text = " ".join(full_transcription)
        return jsonify({"transcript": final_text.strip()})

    except Exception as ex:
        return jsonify({"error": f"Internal server error: {str(ex)}"}), 500
    finally:
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.remove(wav_path)

if __name__ == "__main__":
    app.run(debug=True)
