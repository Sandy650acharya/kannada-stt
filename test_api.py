import requests
import sounddevice as sd
import soundfile as sf
import tempfile
import time
import os
import webbrowser
import numpy as np
import scipy.signal
from pydub import AudioSegment, silence

# CONFIGURATION
TOTAL_DURATION = 120  # Total recording time in seconds
SAMPLE_RATE = 16000
SERVER_URL = "https://kannada-stt.onrender.com/transcribe"

def apply_highpass_filter(audio, samplerate, cutoff=100.0):
    """Apply a basic high-pass filter to reduce low-frequency noise."""
    b, a = scipy.signal.butter(1, cutoff / (0.5 * samplerate), btype='high', analog=False)
    filtered_audio = scipy.signal.filtfilt(b, a, audio[:, 0])
    return np.expand_dims(filtered_audio, axis=1)

def record_full_audio(duration=TOTAL_DURATION, samplerate=SAMPLE_RATE):
    print(f"🎙️ Recording full session for {duration} seconds... Speak continuously.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()

    # Denoise
    audio = apply_highpass_filter(audio, samplerate)

    # Save to temp WAV
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_wav.name, audio, samplerate)
    print(f"✅ Full audio saved to {temp_wav.name}")
    return temp_wav.name

def chunk_audio_by_silence(audio_path):
    print("🔍 Splitting audio based on silence...")
    audio = AudioSegment.from_wav(audio_path).set_channels(1).set_frame_rate(16000)

    chunks = silence.split_on_silence(
        audio,
        min_silence_len=1000,
        silence_thresh=audio.dBFS - 14,
        keep_silence=300
    )

    valid_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunk) < 1000:
            continue
        if len(chunk) <= 30000:
            valid_chunks.append(chunk)
        else:
            # Fallback: split large chunk into ~28s segments
            for j in range(0, len(chunk), 28000):
                valid_chunks.append(chunk[j:j+28000])

    print(f"✅ Total valid chunks: {len(valid_chunks)}")
    return valid_chunks

def send_chunk_to_server(chunk_audio, chunk_index):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
        chunk_audio.export(chunk_file.name, format="wav")
        chunk_path = chunk_file.name

    print(f"📡 Sending chunk {chunk_index} to server...")

    try:
        with open(chunk_path, "rb") as f:
            files = {"audio": f}
            response = requests.post(SERVER_URL, files=files)

        os.remove(chunk_path)

        if response.ok:
            result = response.json()
            transcript = result.get("transcript", "").strip()
            print(f"✅ Chunk {chunk_index} transcript:\n{transcript}\n")
            return transcript
        else:
            print(f"❌ Server error [{response.status_code}]: {response.text}")
            return ""
    except Exception as e:
        print(f"❌ Error sending chunk {chunk_index}: {e}")
        return ""

if __name__ == "__main__":
    audio_path = record_full_audio()
    chunks = chunk_audio_by_silence(audio_path)

    all_transcripts = []
    for idx, chunk in enumerate(chunks, 1):
        transcript = send_chunk_to_server(chunk, idx)
        all_transcripts.append(transcript)

    full_transcript = "\n".join([t for t in all_transcripts if t])
    if full_transcript:
        print("\n📄 Final Combined Transcript:\n")
        print(full_transcript)

        transcript_file = "full_transcript.txt"
        with open(transcript_file, "w", encoding="utf-8") as tf:
            tf.write(full_transcript)
        print(f"📝 Transcript saved to {transcript_file}")
        webbrowser.open(f"file://{os.path.abspath(transcript_file)}")
    else:
        print("⚠️ No valid transcript received.")
