from pydub import AudioSegment
import yt_dlp
from pathlib import Path
import pydub
import whisper
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
import os
import numpy as np

import streamlit as st

accent_id={0:'england' , 1:'us' , 2:'canada' , 3:'australia' ,
           4:'indian' , 5:'scotland' , 6:'ireland' , 7:'african' ,
           8:'malaysia' , 9:'newzealand' , 10:'southatlandtic' , 11:'bermuda' , 
           12:'philippines' , 13:'hongkong' ,14:'wales' ,15:'singapore' }

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
st.write('Whispper MODEL loaded successfully')
classifier = EncoderClassifier.from_hparams(source="Jzuluaga/accent-id-commonaccent_ecapa", savedir="pretrained_models/accent-id-commonaccent_ecapa",local_strategy=LocalStrategy.COPY_SKIP_CACHE)
st.write('Classifire MODEL loaded successfully')
print('Classifire MODEL loaded successfully')
output_wav_file = "output_audio.wav"
try:
    st.write('Extracting the audio from your link ...')
    extract_audio_from_video_link(url, output_wav_file)
    print('Audio loaded successfully')
    st.write('Audio loaded successfully')
    audio_file = AudioSegment.from_file("./temp_video.mp3")
    audio = whisper.load_audio("temp_video.mp3")
    audio_trimed = whisper.pad_or_trim(audio)
    st.write('Analyzing Language spoken')
    mel = whisper.log_mel_spectrogram(audio_trimed).to(model.device)
    _, probs = model.detect_language(mel)
    st.write(f"Detected language: {max(probs, key=probs.get)}")
    st.write('Analyzing Accent spoken using simple method by taking the first 30 seconds')
    audio_trimed.export("output_audio.wav", format="wav")
    out_prob, score, index, text_lab = classifier.classify_file("output_audio.wav")
    st.write(f"Regular Method Detected Accent is: {text_lab} with confidence of {score[0]*100:.2f}%")
    st.write('Analyzing Accent spoken using extensive segmentation method')
    res=model.transcribe(audio,word_timestamps=True)
    probas=np.zeros(len(accent_id.keys()))
    for j in range(len(res['segments'])):
        if res['segments'][j]['no_speech_prob']>0.3:
            continue
        start_ms=res['segments'][j]['start']*1000
        end_ms=res['segments'][j]['end']*1000+500
        segment = audio_file[start_ms:end_ms]
        output_file="./utterances/utterance{}-{}.wav".format(start_ms//1,end_ms//1)
        segment.export(output_file, format="wav")
        out_prob, score, index, text_lab = classifier.classify_file(output_file)
        print(text_lab)
        print(score)
        os.remove(output_file)
        out_prob=out_prob/out_prob.sum()
        probas+=np.asarray(out_prob).reshape(-1)
    probas/=len(res['segments'])
    probas
    #print(f'{accent_id[probas.argmax()]} accent detected with confidence of {probas[probas.argmax()]*100:.2f} % over 15 other english accents')
    
    os.remove("output_audio.wav")
    os.remove("temp_video.mp3")
    st.write(f"Segmentation method Detected Accent is: {accent_id[probas.argmax()]} with confidence of {probas[probas.argmax()]*100:.2f} % over 15 other english accents")

except :
    pass

