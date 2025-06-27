from flask import Flask, request, jsonify, render_template, send_file
import wave
import pyaudio
import threading
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import requests
import re, os
from groq import Groq

app = Flask(__name__)

# Global variables for recording control
is_recording = False
audio_frames = []
transcription_result = ""

# Groq API details
groq_api_key = "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"
groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
groq_model = "llama3-8b-8192"

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

    wf = wave.open("recorded_audio.wav", "wb")
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

#@app.route('/process_audio', methods=['POST'])
def clean_repetitions(text):
    # Use regex to replace consecutive repeated words
    return re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
    
@app.route('/process_audio', methods=['POST'])
def process_audio():
    global transcription_result

    audio = AudioSegment.from_wav("recorded_audio.wav")
    silence_thresh = audio.dBFS - 14  # Dynamically adjust based on the average volume
    chunks = split_on_silence(audio, min_silence_len=200, silence_thresh=silence_thresh)

    #chunks = split_on_silence(audio, min_silence_len=200, silence_thresh=-35)

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

    output_audio.export("processed_audio.wav", format="wav")
    client = Groq(api_key="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")
    filename = "processed_audio.wav"
    try:
        with open(filename, "rb") as file:
        # Create a transcription of the audio file
            transcription_result = client.audio.transcriptions.create(
                file=(filename, file.read()),  # Required audio file
                model="distil-whisper-large-v3-en",  # Required model to use for transcription
                response_format="json",  # Optional
                language="en",  # Optional
                temperature=0.0  # Optional
            )
        # Clean up consecutive repeated words
        transcription_result = clean_repetitions(transcription_result)
        return jsonify({"transcription": transcription_result})
    except sr.UnknownValueError:
        return jsonify({"error": "Speech recognition could not understand the audio."})
    except sr.RequestError as e:
        return jsonify({"error": f"Could not request results from the service; {e}"})


''' recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.0
    with sr.AudioFile("processed_audio.wav") as source:
        audio_data = recognizer.record(source)
        try:
            transcription_result = recognizer.recognize_google(audio_data)
            # Clean up consecutive repeated words
            transcription_result = clean_repetitions(transcription_result)
            return jsonify({"transcription": transcription_result})
        except sr.UnknownValueError:
            return jsonify({"error": "Speech recognition could not understand the audio."})
        except sr.RequestError as e:
            return jsonify({"error": f"Could not request results from the service; {e}"})'''


@app.route('/generate_response', methods=['POST'])
def generate_response():
    data = request.get_json()
    transcription = data.get("transcription", "")

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": groq_model,
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

    response = requests.post(groq_api_url, json=payload, headers=headers)
    if response.status_code == 200:
        groq_response = response.json()
        response_text = groq_response['choices'][0]['message']['content']

        tts = gTTS(text=response_text, lang='en')
        tts.save("response_audio.mp3")
        return jsonify({"response_text": response_text, "audio_file": "response_audio.mp3"})
    else:
        return jsonify({"error": f"API error: {response.status_code}, {response.text}"})


@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_file(filename, as_attachment=True)

@app.route('/')
def index():
    return render_template('index.html')  # HTML template for basic UI

if __name__ == '__main__':
    app.run(debug=True)

