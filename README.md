# Accent recognition from raw speech 

`Accent Identification from Speech Recordings combining segmenteation obtained from [whisper](https://github.com/openai/whisper)  with ECAPA-TDNN embeddings method from [CommonAccent (CV 11.0) recognition Repository](https://github.com/JuanPZuluaga/accent-recog-slt2022/tree/main)

<p align="center">
    <a href="https://github.com/mohd-alhussin/Accent_detection/">
</p>

# Accent recognition from raw speech 

Method we use whisper to segment the audio file into speech region 
then we use off the shelf method to detect accent
**Accents avialable**: 

<accent> <id>
-----------------------------

<accent-id> <duration-in-hrs>
-----------------------------
'england' => 0
'us' => 1
'canada' => 2
'australia' => 3
'indian' => 4
'scotland' => 5
'ireland' => 6
'african' => 7
'malaysia' => 8
'newzealand' => 9
'southatlandtic' => 10
'bermuda' => 11
'philippines' => 12
'hongkong' => 13
'wales' => 14
'singapore' => 15

```python
# Get started: 

Step 1: Using python 3.9: install python and the requirements

```bash
python -m pip install -r requirements.txt
```




# Run 
python detect_accent.py

# Streamlit Demo 
## Locally:
streamlit run .\detect_accent.py

## live:
https://accentdetection-mulla.streamlit.app/
or 
https://accentdetection-maze.streamlit.app/



The results of this project will be submitted to Interspeech 2023. 
