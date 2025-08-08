# app.py
from flask import Flask, request, jsonify
import speech_recognition as sr
import tempfile
from pydub import AudioSegment, silence
import os
import re

# Optional punctuation model (English only)
try:
    from deepsegment import DeepSegment
    segmenter = DeepSegment("en")
    print("[INFO] DeepSegment loaded for English punctuation.")
except Exception:
    segmenter = None
    print("[INFO] DeepSegment NOT available — punctuation disabled for English.")

app = Flask(__name__)


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def merge_transcripts_with_dedup(transcripts, max_overlap_words=3):
    """
    Join chunk transcripts with a simple overlap-deduplication heuristic.
    Looks for an overlap of up to max_overlap_words between the end of the previous
    transcript and the start of the next transcript and removes duplicates.
    Returns a single merged string.
    """
    if not transcripts:
        return ""

    merged = transcripts[0].strip()
    for nxt in transcripts[1:]:
        prev_words = merged.split()
        next_words = nxt.strip().split()
        overlap_found = 0

        # test overlaps length from max_overlap_words down to 1
        for k in range(max_overlap_words, 0, -1):
            if len(prev_words) >= k and len(next_words) >= k:
                if prev_words[-k:] == next_words[:k]:
                    overlap_found = k
                    break

        if overlap_found:
            # drop the overlapping prefix from next_words
            merged += " " + " ".join(next_words[overlap_found:])
        else:
            merged += " " + " ".join(next_words)

    return normalize_whitespace(merged)


@app.route("/")
def index():
    return "✅ Multilingual STT API is running!"


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """
    POST /transcribe?lang=kn-IN&punctuate=1
    body: form-data audio: <wav file>

    Returns JSON:
      {
        "transcript": "raw merged transcript",
        "punctuated": "optional punctuated transcript"  # only if punctuation applied
      }
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    # Language parameter (default to Kannada)
    lang_code = request.args.get("lang", "kn-IN").strip() or "kn-IN"

    # Whether to try punctuation (also auto-enabled for English if segmenter present)
    punctuate_flag = request.args.get("punctuate", "").strip()
    punctuate = bool(punctuate_flag) or (lang_code.startswith("en") and segmenter is not None)

    audio_file = request.files["audio"]
    wav_path = None
    processed_chunks = []

    try:
        # 1) Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_file.save(tmp.name)
            wav_path = tmp.name

        # 2) Load and normalize audio
        audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(16000)

        # 3) Silence-based chunking (short pauses ~ natural speech)
        primary_chunks = silence.split_on_silence(
            audio,
            min_silence_len=800,  # 0.8s silence threshold
            silence_thresh=audio.dBFS - 14,
            keep_silence=400,  # small padding to keep context
        )

        # 4) Merge into chunks targeting ~20-30s with small overlap to avoid truncation
        max_chunk_ms = 30000  # 30 seconds
        overlap_ms = 500      # 0.5s overlap
        current = AudioSegment.silent(duration=0)

        for idx, chunk in enumerate(primary_chunks):
            if len(chunk) < 800:  # skip extremely tiny noises
                continue

            if len(current) + len(chunk) <= max_chunk_ms:
                current += chunk
            else:
                if len(current) > 800:
                    processed_chunks.append(current)
                current = chunk

            # add overlap padding between pieces to reduce word-cut risk
            if idx < len(primary_chunks) - 1:
                current += AudioSegment.silent(duration=overlap_ms)

        if len(current) > 800:
            processed_chunks.append(current)

        # If no chunks detected, fallback to treating entire audio as one chunk
        if not processed_chunks:
            processed_chunks = [audio]

        # 5) Transcribe each chunk with Google STT
        recognizer = sr.Recognizer()
        raw_transcripts = []

        for i, chunk in enumerate(processed_chunks, start=1):
            # export chunk to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
                chunk.export(chunk_file.name, format="wav")
                chunk_path = chunk_file.name

            try:
                with sr.AudioFile(chunk_path) as source:
                    # short ambient adjustment; keep small to avoid long processing time
                    try:
                        recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    except Exception:
                        # If ambient adjustment fails for some reason, continue
                        pass
                    audio_data = recognizer.record(source)

                # call Google STT
                text = recognizer.recognize_google(audio_data, language=lang_code)
                text = normalize_whitespace(text)
                raw_transcripts.append(text)

            except sr.UnknownValueError:
                # record unknown as empty string (do not pollute punctuation)
                raw_transcripts.append("")
            except sr.RequestError as e:
                # Google API/network issue -> abort with 500
                return jsonify({"error": f"Google API error: {str(e)}"}), 500
            finally:
                # cleanup chunk file
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass

        # 6) Merge transcripts with dedup heuristic and clean placeholders
        # remove empty segments and placeholders
        cleaned = [t for t in raw_transcripts if t and t.strip()]
        merged = merge_transcripts_with_dedup(cleaned, max_overlap_words=3)

        # final cleanup (strip repeated whitespace)
        merged = normalize_whitespace(merged)

        response_payload = {"transcript": merged}

        # 7) Optional punctuation for English using DeepSegment (or if user asked)
        if punctuate and lang_code.startswith("en") and segmenter:
            try:
                # DeepSegment expects raw text -> returns list of sentences
                segmented = segmenter.segment_long(merged)
                # join sentences with space — segmented items should include punctuation
                punctuated = " ".join([s.strip() for s in segmented if s and s.strip()])
                punctuated = normalize_whitespace(punctuated)
                # If DeepSegment produced no punctuation (same as input), still return it as 'punctuated'
                response_payload["punctuated"] = punctuated
            except Exception as e:
                # fail gracefully, include debug note
                print(f"[⚠️ DeepSegment error]: {e}")
                response_payload["punctuation_error"] = str(e)

        return jsonify(response_payload)

    except Exception as ex:
        # broad fallback
        return jsonify({"error": f"Internal server error: {str(ex)}"}), 500

    finally:
        # always remove temp upload
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


if __name__ == "__main__":
    app.run(debug=True)
