from flask import Flask, request, jsonify, render_template, send_file
import wave
import pyaudio
import threading
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import requests
import re
import os
from gtts import gTTS

app = Flask(__name__)

# Global variables for recording control
is_recording = False
audio_frames = []
transcription_result = ""

# Groq API configuration
groq_api_key = ""
groq_model = "distil-whisper-large-v3-en"

# Audio recording function
def record_audio():
    global is_recording, audio_frames
    audio_frames = []
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    while is_recording:
        data = stream.read(1024)
        audio_frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save recorded audio in 'static' folder
    file_path = os.path.join('static', 'recorded_audio.wav')
    wf = wave.open(file_path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(audio_frames))
    wf.close()

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    global is_recording
    if is_recording:
        is_recording = False
    else:
        is_recording = True
        threading.Thread(target=record_audio).start()
    return jsonify({"status": "recording" if is_recording else "stopped"})

def clean_repetitions(text):
    return re.sub(r'\b(\w+)\s+\1\b', r'\1', text)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    global transcription_result

    # Load and process audio from static folder
    file_path = os.path.join('static', 'recorded_audio.wav')
    audio = AudioSegment.from_wav(file_path)
    silence_thresh = audio.dBFS - 14
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=silence_thresh)

    def compare_chunks(chunk1, chunk2, threshold=0.8):
        arr1 = np.array(chunk1.get_array_of_samples())
        arr2 = np.array(chunk2.get_array_of_samples())
        min_len = min(len(arr1), len(arr2))
        arr1, arr2 = arr1[:min_len], arr2[:min_len]
        norm1, norm2 = arr1 / np.max(np.abs(arr1)), arr2 / np.max(np.abs(arr2))
        similarity = np.corrcoef(norm1, norm2)[0, 1]
        return similarity > threshold

    def detect_stutter(chunks):
        stutter_free_chunks = []
        previous_chunk = None
        for chunk in chunks:
            if previous_chunk and compare_chunks(chunk, previous_chunk):
                continue
            stutter_free_chunks.append(chunk)
            previous_chunk = chunk
        return stutter_free_chunks

    stutter_free_chunks = detect_stutter(chunks)
    output_audio = AudioSegment.silent(duration=0)
    for chunk in stutter_free_chunks:
        output_audio += chunk
    output_audio += AudioSegment.silent(duration=500)

    # Save processed audio in 'static' folder
    processed_audio_path = os.path.join('static', 'processed_audio.wav')
    output_audio.export(processed_audio_path, format="wav")

    # Use Groq API for transcription
    with open(processed_audio_path, "rb") as file:
        response = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {groq_api_key}"},
            files={"file": file},
            data={
                "model": groq_model,
                "response_format": "json",
                "language": "en",
                "temperature": "0.0"
            }
        )
        if response.status_code == 200:
            transcription_data = response.json()
            transcription_result = transcription_data.get("text", "")
            transcription_result = clean_repetitions(transcription_result)
            return jsonify({"transcription": transcription_result})
        else:
            return jsonify({"error": f"Groq API error: {response.status_code}, {response.text}"})


@app.route('/generate_response', methods=['POST'])
def generate_response():
    data = request.get_json()
    transcription = data.get("transcription", "")

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",  # Chat completion-specific model
        "messages": [
            {
                "role": "system",
                "content": "Please limit the response to 80 words or fewer."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    }

    # Call Groq API for chat completion
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
    if response.status_code == 200:
        groq_response = response.json()
        response_text = groq_response['choices'][0]['message']['content']

        # Convert text response to speech using gTTS
        tts = gTTS(text=response_text, lang='en')
        audio_file_path = os.path.join('static', 'response_audio.mp3')  # Save in 'static' folder
        tts.save(audio_file_path)

        # Return response JSON with path to the saved audio file
        return jsonify({"response_text": response_text, "audio_file": "response_audio.mp3"})
    else:
        return jsonify({"error": f"API error: {response.status_code}, {response.text}"})


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
