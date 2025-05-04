# AI-Generated Voice Detection and Speech Recognition

A Streamlit web application for detecting AI-generated speech and performing speech-to-text transcription with optional translation.

---

## Features

* **AI Voice Detection**

  * Upload MP3 or WAV audio files
  * Visualize the audio waveform
  * Classify the probability that the audio is AI-generated using a pretrained `AudioMiniEncoderWithClassifierHead` model
  * Display the likelihood (%) of AI generation

* **Speech Recognition & Translation**

  * Upload MP3 or WAV audio files
  * Transcribe speech to text using Google’s Speech-to-Text API via the `speech_recognition` library
  * Translate the transcript into target languages (Telugu, French, Spanish, German) using the `deep_translator` package

* **Interactive UI**

  * Two tabs for separate workflows
  * Real-time audio playback and waveform plotting
  * Informative messages, warnings, and error handling

---

## Usage

### AI Voice Detection

1. Select the **AI Voice Detection** tab.
2. Upload an `.mp3` or `.wav` audio file.
3. Click **Analyze Audio**.
4. View the waveform plot and the probability that the clip is AI-generated.

### Speech Recognition

1. Select the **Speech Recognition** tab.
2. Upload an `.mp3` or `.wav` audio file.
3. Click **Transcribe Audio**.
4. The recognized text will appear on screen.
5. Choose a target language from the dropdown to see the machine translation.

---

## Requirements

* streamlit
* librosa
* torch
* torchaudio
* plotly
* speechrecognition
* pydub
* deep\_translator
* tortoise-tts

---

## Model Details

* **Audio Classifier**

  * Architecture: `AudioMiniEncoderWithClassifierHead` from Tortoise TTS
  * Input: single-channel, 22 kHz audio
  * Output: 2-class softmax probability

* **Speech Recognition**

  * Uses Google Web Speech API (requires an active internet connection)

---

## Disclaimer

This tool provides heuristic signals and is **not** 100% accurate. Use results as a strong indicator—not a definitive verdict.

---

## Acknowledgments

* [Streamlit](https://streamlit.io/) for the web framework
* [Tortoise TTS](https://github.com/neonbjb/tortoise-tts) for the audio classifier
* [Google Speech-to-Text API](https://cloud.google.com/speech-to-text)
* [Deep Translator](https://pypi.org/project/deep-translator/) for translation services
