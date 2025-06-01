from pydub import AudioSegment
import yt_dlp
from pathlib import Path
import pydub
import whisper
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
import os

import streamlit as st



def extract_audio_from_video_link(video_link, output_audio_path):
    ydl_opts = {
       'extract_audio': True, 'format': 'bestaudio',
       
       'outtmpl': 'temp_video.mp3',
   }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_link])
    sound = AudioSegment.from_file("./temp_video.mp3")
    sound.export(output_audio_path, format="wav")


st.title("Spoken Accent Detection")
st.write("Please Enter the URL of the video you want to analyse")
url = st.text_input('The URL link')

model = whisper.load_model("tiny")
print('Whispper MODEL loaded successfully')
classifier = EncoderClassifier.from_hparams(source="Jzuluaga/accent-id-commonaccent_ecapa", savedir="pretrained_models/accent-id-commonaccent_ecapa",local_strategy=LocalStrategy.COPY_SKIP_CACHE)
print('Whispper MODEL loaded successfully')
output_wav_file = "output_audio.wav"
try:
    extract_audio_from_video_link(url, output_wav_file)
    print('Audio downloaded successfully')

    audio_file = AudioSegment.from_file("./temp_video.mp3")
    

    
    audio = whisper.load_audio("temp_video.mp3")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    st.write(f"Detected language: {max(probs, key=probs.get)}")


    
    out_prob, score, index, text_lab = classifier.classify_file("output_audio.wav")
    print(text_lab)
    print(score)
    st.write(f"Detected Accent is: {text_lab} with confidence of {score[0]*100:.2f}%")
except :
    pass


