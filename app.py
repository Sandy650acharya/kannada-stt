from flask import Flask, request, jsonify
import speech_recognition as sr
import os

app = Flask(__name__)
recognizer = sr.Recognizer()

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']
    audio_path = "temp.wav"
    audio_file.save(audio_path)

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language="kn-IN")
            os.remove(audio_path)
            return jsonify({'transcript': text})
        except Exception as e:
            os.remove(audio_path)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
