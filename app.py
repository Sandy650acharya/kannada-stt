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
    wav_path = None
    processed_chunks = []

    try:
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            audio_file.save(temp.name)
            wav_path = temp.name

        # Load and preprocess
        audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(16000)

        # Initial silence-based split
        primary_chunks = silence.split_on_silence(
            audio,
            min_silence_len=1000,
            silence_thresh=audio.dBFS - 14,
            keep_silence=300
        )

        # Filter out short chunks and re-split long ones (>30s)
        for chunk in primary_chunks:
            if len(chunk) < 1000:
                continue  # Skip super short noise

            if len(chunk) <= 30000:  # 30 sec max
                processed_chunks.append(chunk)
            else:
                # Re-split overly long chunk into smaller ones (fallback)
                sub_chunks = [chunk[i:i + 28000] for i in range(0, len(chunk), 28000)]
                processed_chunks.extend(sub_chunks)

        if not processed_chunks:
            return jsonify({"error": "No valid speech detected after preprocessing."}), 400

        # Transcribe each chunk
        recognizer = sr.Recognizer()
        transcripts = []

        for i, chunk in enumerate(processed_chunks):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
                chunk.export(chunk_file.name, format="wav")
                chunk_path = chunk_file.name

            try:
                with sr.AudioFile(chunk_path) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio_data = recognizer.record(source)

                text = recognizer.recognize_google(audio_data, language="kn-IN")
                transcripts.append(text)

            except sr.UnknownValueError:
                transcripts.append("[Unrecognized]")
            except sr.RequestError as e:
                return jsonify({"error": f"Google API error: {str(e)}"}), 500
            finally:
                os.remove(chunk_path)

        final_transcript = " ".join(transcripts).strip()
        return jsonify({"transcript": final_transcript})

    except Exception as ex:
        return jsonify({"error": f"Internal server error: {str(ex)}"}), 500
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

if __name__ == "__main__":
    app.run(debug=True)
