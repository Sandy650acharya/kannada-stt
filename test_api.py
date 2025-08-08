import requests
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import webbrowser
import numpy as np
import scipy.signal
from pydub import AudioSegment, silence
import re

# CONFIGURATION
TOTAL_DURATION = 30   # Max recording time (seconds)
SAMPLE_RATE = 16000
SERVER_URL = "https://kannada-stt.onrender.com/transcribe"
AUTO_SAVE_TRANSCRIPT = True


def choose_language():
    print("\n🌐 Choose Language for Transcription:")
    print("1. Kannada (Default)")
    print("2. English")
    choice = input("Enter choice (1 or 2): ").strip()
    return "en-IN" if choice == "2" else "kn-IN"


def apply_highpass_filter(audio, samplerate, cutoff=100.0):
    """Apply a high-pass filter to reduce low-frequency noise."""
    b, a = scipy.signal.butter(1, cutoff / (0.5 * samplerate), btype="high", analog=False)
    filtered_audio = scipy.signal.filtfilt(b, a, audio[:, 0])
    return np.expand_dims(filtered_audio, axis=1)


def record_full_audio(duration=TOTAL_DURATION, samplerate=SAMPLE_RATE):
    print(f"\n🎙️ Recording for {duration} seconds... Speak clearly.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    audio = apply_highpass_filter(audio, samplerate)

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_wav.name, audio, samplerate)
    print(f"✅ Audio saved at {temp_wav.name}")
    return temp_wav.name


def chunk_audio_by_silence(audio_path):
    print("\n🔍 Splitting audio using silence detection...")
    audio = AudioSegment.from_wav(audio_path).set_channels(1).set_frame_rate(16000)

    primary_chunks = silence.split_on_silence(
        audio,
        min_silence_len=800,
        silence_thresh=audio.dBFS - 14,
        keep_silence=300
    )

    valid_chunks = []
    current = AudioSegment.silent(duration=0)
    overlap = AudioSegment.silent(duration=500)  # 0.5s overlap

    for chunk in primary_chunks:
        if len(chunk) < 1000:
            continue
        if len(current) + len(chunk) <= 30000:
            current += chunk + overlap
        else:
            if len(current) > 1000:
                valid_chunks.append(current)
            current = chunk

    if len(current) > 1000:
        valid_chunks.append(current)

    print(f"✅ Created {len(valid_chunks)} valid chunks.")
    return valid_chunks


def merge_transcripts_with_dedup(transcripts, max_overlap_words=3):
    """Merge transcripts while removing repeated overlap at boundaries."""
    if not transcripts:
        return ""
    merged = transcripts[0].strip()
    for nxt in transcripts[1:]:
        prev_words = merged.split()
        next_words = nxt.strip().split()
        overlap_found = 0
        for k in range(max_overlap_words, 0, -1):
            if len(prev_words) >= k and len(next_words) >= k:
                if prev_words[-k:] == next_words[:k]:
                    overlap_found = k
                    break
        if overlap_found:
            merged += " " + " ".join(next_words[overlap_found:])
        else:
            merged += " " + " ".join(next_words)
    return re.sub(r"\s+", " ", merged).strip()


def send_chunk_to_server(chunk_audio, chunk_index, language):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
        chunk_audio.export(chunk_file.name, format="wav")
        chunk_path = chunk_file.name

    print(f"📡 Sending chunk {chunk_index} to server...")

    try:
        punct_param = "&punctuate=1" if language.startswith("en") else ""
        with open(chunk_path, "rb") as f:
            response = requests.post(
                f"{SERVER_URL}?lang={language}{punct_param}",
                files={"audio": f}
            )
        os.remove(chunk_path)

        if response.ok:
            result = response.json()
            transcript = result.get("punctuated") or result.get("transcript", "")
            transcript = transcript.strip()
            print(f"✅ Chunk {chunk_index} transcript:\n{transcript}\n")
            return transcript
        else:
            print(f"❌ Server error [{response.status_code}]: {response.text}")
            return ""
    except Exception as e:
        print(f"❌ Error sending chunk {chunk_index}: {e}")
        return ""


def save_transcript(text, filename="full_transcript.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as tf:
            tf.write(text)
        print(f"📝 Transcript saved to {filename}")
        webbrowser.open(f"file://{os.path.abspath(filename)}")
    except Exception as e:
        print(f"⚠️ Could not save/open transcript: {e}")


if __name__ == "__main__":
    selected_language = choose_language()
    audio_path = record_full_audio()
    chunks = chunk_audio_by_silence(audio_path)

    all_transcripts = []
    for idx, chunk in enumerate(chunks, 1):
        transcript = send_chunk_to_server(chunk, idx, selected_language)
        if transcript:
            all_transcripts.append(transcript)

    # Merge with deduplication
    full_transcript = merge_transcripts_with_dedup(all_transcripts)

    if full_transcript:
        print("\n📄 Final Combined Transcript:\n")
        print(full_transcript)
        if AUTO_SAVE_TRANSCRIPT:
            save_transcript(full_transcript)
    else:
        print("⚠️ No valid transcript received.")
