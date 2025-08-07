from flask import Flask, request, jsonify
import speech_recognition as sr
import tempfile
from pydub import AudioSegment, silence
import os

# Optional punctuation model (English only)
try:
    from deepsegment import DeepSegment
    segmenter = DeepSegment('en')
except ImportError:
    segmenter = None  # Fallback if DeepSegment not installed

app = Flask(__name__)

@app.route("/")
def index():
    return "✅ Multilingual STT API is running!"

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    # Handle language preference (default: Kannada)
    lang_code = request.args.get("lang", "kn-IN").strip()
    if not lang_code:
        lang_code = "kn-IN"

    audio_file = request.files["audio"]
    wav_path = None
    processed_chunks = []

    try:
        # 1️⃣ Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            audio_file.save(temp.name)
            wav_path = temp.name

        # 2️⃣ Load and normalize audio
        audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(16000)

        # 3️⃣ Split based on silence
        primary_chunks = silence.split_on_silence(
            audio,
            min_silence_len=800,              # Detect natural pauses
            silence_thresh=audio.dBFS - 14,   # Adaptive threshold
            keep_silence=400                  # Context padding
        )

        # 4️⃣ Merge chunks with overlap (~30s max each)
        max_chunk_len = 30000      # 30 sec max
        overlap_ms = 500           # Add 0.5s overlap to avoid cut words
        current = AudioSegment.silent(duration=0)

        for i, chunk in enumerate(primary_chunks):
            if len(chunk) < 1000:
                continue

            if len(current) + len(chunk) <= max_chunk_len:
                current += chunk
            else:
                if len(current) > 1000:
                    processed_chunks.append(current)
                current = chunk

            if i < len(primary_chunks) - 1:
                current += AudioSegment.silent(duration=overlap_ms)

        if len(current) > 1000:
            processed_chunks.append(current)

        if not processed_chunks:
            return jsonify({"error": "No valid speech detected after processing."}), 400

        # 5️⃣ Transcribe each chunk using Google STT
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

                text = recognizer.recognize_google(audio_data, language=lang_code)
                transcripts.append(text)

            except sr.UnknownValueError:
                transcripts.append("[Unrecognized]")
            except sr.RequestError as e:
                return jsonify({"error": f"Google API error: {str(e)}"}), 500
            finally:
                os.remove(chunk_path)

        # 6️⃣ Combine all and apply punctuation (English only)
        final_transcript = " ".join(transcripts).strip()

        if lang_code.startswith("en") and segmenter:
            try:
                final_transcript = " ".join(segmenter.segment_long(final_transcript))
            except Exception as e:
                # Log or fallback gracefully if segmentation fails
                print(f"[⚠️ DeepSegment error]: {e}")

        return jsonify({"transcript": final_transcript})

    except Exception as ex:
        return jsonify({"error": f"Internal server error: {str(ex)}"}), 500
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

if __name__ == "__main__":
    app.run(debug=True)
