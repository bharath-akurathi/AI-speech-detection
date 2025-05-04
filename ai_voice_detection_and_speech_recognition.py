import streamlit as st
import os
import io
import librosa
import plotly.express as px
import torch
import torch.nn.functional as F
import torchaudio
import speech_recognition as sr
from deep_translator import GoogleTranslator
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
from pydub import AudioSegment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_audio(audiopath, sampling_rate=22000):
    try:
        if isinstance(audiopath, str):
            if audiopath.endswith('.mp3'):
                audio, lsr = librosa.load(audiopath, sr=sampling_rate)
                audio = torch.FloatTensor(audio)
            else:
                raise ValueError(f"Unsupported audio format provided: {audiopath[-4:]}")
        elif isinstance(audiopath, io.BytesIO):
            audio, lsr = torchaudio.load(audiopath)
            audio = audio[0]
        else:
            raise TypeError("Unsupported audio input type")

        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

        audio = audio.clip(-1, 1)
        return audio.unsqueeze(0)
    except Exception as e:
        st.error(f"Error loading audio: {str(e)}")
        return None

def classify_audio_clip(clip):
    try:
        classifier = AudioMiniEncoderWithClassifierHead(
            classes=2,
            spec_dim=1,
            embedding_dim=512,
            depth=5,
            downsample_factor=4,
            resnet_blocks=2,
            attn_blocks=4,
            num_attn_heads=4,
            base_channels=32,
            dropout=0,
            kernel_size=5
        )

        state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
        classifier.load_state_dict(state_dict)
        classifier.eval()

        clip = clip.cpu().unsqueeze(0)
        results = F.softmax(classifier(clip), dim=-1)
        return results[0][0].item()
    except Exception as e:
        st.error(f"Error classifying audio: {str(e)}")
        return None

def convert_to_wav(uploaded_file):
    try:
        audio = AudioSegment.from_file(uploaded_file)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)  # Move to the beginning of the BytesIO object
        return wav_io
    except Exception as e:
        st.error(f"Error converting audio to WAV: {str(e)}")
        return None

def transcribe_audio(uploaded_file):
    try:
        recognizer = sr.Recognizer()
        # Convert to WAV format
        wav_file = convert_to_wav(uploaded_file)
        if wav_file is None:
            return None
        
        with sr.AudioFile(wav_file) as source:
            audio = recognizer.record(source)
            result = recognizer.recognize_google(audio, language="en")
            return result
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

def translate_text(text, target_language='te'):
    try:
        translator = GoogleTranslator(target=target_language)
        translation = translator.translate(text)
        return translation
    except Exception as e:
        st.error(f"Error translating text: {str(e)}")
        return None

st.set_page_config(layout="wide")

def main():
    st.title("AI-Generated Voice Detection and Speech Recognition")
    
    tab1, tab2 = st.tabs(["AI Voice Detection", "Speech Recognition"])

    with tab1:
        uploaded_file_1 = st.file_uploader("Upload an audio file for AI Voice Detection", type=["mp3", "wav"], key="audio_detection_uploader")
        
        if uploaded_file_1 is not None:
            if st.button("Analyze Audio"):
                col1, col2, col3 = st.columns(3)
                
                with col2:
                    st.info("Your results are below")
                    audio_clip = load_audio(uploaded_file_1)
                    if audio_clip is not None:
                        result = classify_audio_clip(audio_clip)
                        result1 = f'{result:.4f}'
                        if result1 and result is not None:
                            if result1 == '0.0000':
                                result = 1
                            st.info(f"Result Probability: {result:.4f}")
                            st.success(f"The uploaded audio is {result * 100:.2f}% likely to be AI Generated.")
                        else:
                            st.error("Failed to classify the audio.")
                    else:
                        st.error("Failed to load the audio.")
                
                with col1:
                    st.info("Your uploaded audio is below")
                    st.audio(uploaded_file_1)
                    fig = px.line(x=list(range(len(audio_clip.squeeze()))), y=audio_clip.squeeze())
                    fig.update_layout(
                        title="Waveform Plot",
                        xaxis_title="Time",
                        yaxis_title="Amplitude"
                    )
                    st.plotly_chart(fig, use_container_width=True )
                
                with col3:
                    st.info("Disclaimer")
                    st.warning("These classification or detection mechanisms are not always accurate. They should be considered as a strong signal and not the ultimate decision makers.")

    with tab2:
        uploaded_file_2 = st.file_uploader("Upload an audio file for Speech Recognition", type=["mp3", "wav"], key="speech_recognition_uploader")
        
        if uploaded_file_2 is not None:
            if st.button("Transcribe Audio"):
                transcription = transcribe_audio(uploaded_file_2)
                if transcription:
                    st.info("Transcription:")
                    st.write(transcription)
                    
                    target_language = st.selectbox("Select target language", ['te', 'fr', 'es', 'de'])
                    translation = translate_text(transcription, target_language)
                    if translation:
                        st.info("Translation:")
                        st.write(translation)
                else:
                    st.error("Failed to transcribe audio.")

if __name__ == "__main__":
    main()