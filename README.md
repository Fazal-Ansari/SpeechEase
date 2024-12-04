# SpeechEase
# SpeechEase
# 🎙️ SpeechEase Voice Assistant

Welcome to **SpeechEase Voice Assistant**! This is a voice-powered web application built with **Flask**, designed to help individuals with **stuttering** by providing seamless voice-to-text and text-to-speech features. The app allows users to record their audio, transcribe it using the **Groq API**, and receive an AI-generated response. It also converts the response to audio using **gTTS** (Google Text-to-Speech), providing a smooth, interactive experience. 🚀

## 🤝 Why It’s Needed

Stuttering can impact the way individuals express themselves, making communication more difficult, frustrating, or even embarrassing. **SpeechEase** was created to assist people who stutter by offering several key features that can help alleviate these challenges:

- **Stutter Detection**: Automatically detects and removes stutters from recorded audio, helping individuals speak more fluently without worrying about interruptions in their speech.
- **Speech-to-Text**: Converts speech into text with the Groq API, enabling clear transcription even when speech may be affected by stuttering.
- **AI-Generated Responses**: Provides intelligent and responsive answers to questions or statements, offering a real-time, engaging conversation that doesn’t require users to worry about repeating themselves.
- **Text-to-Speech Conversion**: Converts generated text responses into speech, creating a more natural, interactive experience for users.

This project is designed to give **individuals with speech challenges** a better and more comfortable way to communicate. It aims to reduce frustration and enhance confidence when speaking by focusing on fluency and minimizing the impact of stutters.

---

## ✨ Features

- 🎤 **Record Audio**: Capture audio directly from the web interface.
- 📄 **Transcribe Audio**: Convert speech to text using the Groq API.
- 🤖 **Generate AI Responses**: Get concise, AI-generated responses to your queries.
- 🔊 **Text-to-Speech Conversion**: Listen to the AI-generated responses with gTTS.
- 🎛️ **Stutter Detection**: Automatically detect and remove stutters from recorded audio, allowing for smoother communication.

---

## 🛠️ Technologies Used

- **Python** 🐍 (Flask, PyAudio, Pydub, gTTS)
- **HTML, CSS, JavaScript** 🎨 for the frontend
- **Groq API** 🧠 for transcription and chat completions
- **Google Cloud or Local Hosting** ☁️

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/speechease-voice-assistant.git
cd speechease-voice-assistant

📁 speechease-voice-assistant/
├── 📂 static/               # Static assets (CSS, JS, audio files)
├── 📂 templates/            # HTML templates
├── app.py                   # Main Flask app
├── requirements.txt         # Dependencies

## Methodology
🎤 Start Recording --> 🛑 Stop Recording --> 🧹 Clean Up Audio --> 📄 Transcribe Audio
         ↓                         ↓                     ↓                     ↓
   (Recorded Audio)            (Processed Audio)    (Transcription)     (Cleaned Text)
         ↓                         ↓                     ↓                     ↓
🤖 Generate AI Response --> 🔊 Convert Text to Speech --> 🔄 Play Audio Response
         ↓                                        ↓
   (Response Text)                        (Response Audio)

---

📝 To-Do later :
 🔍 Improve transcription accuracy with advanced NLP models.
 📈 Add analytics for tracking user interactions.
 🌐 Support for multiple languages.
 🎨 Enhance UI/UX with custom animations.
