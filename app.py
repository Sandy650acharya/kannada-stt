from flask import Flask, request, jsonify
import speech_recognition as sr
import tempfile
from pydub import AudioSegment, silence
import os

try:
    from deepsegment import DeepSegment
    segmenter = DeepSegment('en')
except ImportError:
    segmenter = None  # DeepSegment only loaded for English

app = Flask(__name__)

@app.route("/")
def index():
    return "✅ Multilingual STT API is running!"

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    lang_code = request.args.get("lang", "kn-IN").strip()
    if not lang_code:
        lang_code = "kn-IN"

    audio_file = request.files["audio"]
    wav_path = None
    processed_chunks = []

    try:
        # Step 1: Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            audio_file.save(temp.name)
            wav_path = temp.name

        # Step 2: Load and normalize audio
        audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(16000)

        # Step 3: Silence-based chunking
        primary_chunks = silence.split_on_silence(
            audio,
            min_silence_len=800,
            silence_thresh=audio.dBFS - 14,
            keep_silence=400  # Add context
        )

        # Step 4: Merge into ~30s chunks with overlap
        max_chunk_len = 30000  # 30 seconds
        overlap_ms = 500       # 0.5s overlap to prevent word truncation

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

            # Add small overlap at end of each segment (except last)
            if i < len(primary_chunks) - 1:
                current += AudioSegment.silent(duration=overlap_ms)

        if len(current) > 1000:
            processed_chunks.append(current)

        if not processed_chunks:
            return jsonify({"error": "No valid speech detected after processing."}), 400

        # Step 5: Transcribe each chunk
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

        # Step 6: Join and optionally punctuate
        final_transcript = " ".join(transcripts).strip()

        # Add punctuation for English using DeepSegment
        if lang_code.startswith("en") and segmenter:
            punctuated = segmenter.segment_long(final_transcript)
            final_transcript = " ".join(punctuated)

        return jsonify({"transcript": final_transcript})

    except Exception as ex:
        return jsonify({"error": f"Internal server error: {str(ex)}"}), 500
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

if __name__ == "__main__":
    app.run(debug=True)
