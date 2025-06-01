# Accent recognition from raw speech 

`Accent Identification from Speech Recordings combining segmenteation obtained from [whisper](https://github.com/openai/whisper)  with ECAPA-TDNN embeddings method from [CommonAccent (CV 11.0) recognition Repository](https://github.com/JuanPZuluaga/accent-recog-slt2022/tree/main)

<p align="center">
    <a href="https://github.com/mohd-alhussin/Accent_detection/">
</p>


**Accents avialable**: 

<accent> <id>
-----------------------------
* Austrian - 104
* East African Khoja - 107
* Dutch - 108
* West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad) - 282
* Welsh English - 623
* Malaysian English - 1004
* Liverpool English,Lancashire English,England English - 2571
* Singaporean English - 2792
* Hong Kong English - 2951
* Filipino - 4030
* Southern African (South Africa, Zimbabwe, Namibia) - 4270
* New Zealand English - 4960
* Irish English - 6339
* Northern Irish - 6862
* Scottish English - 10817
* Australian English - 33335
* German English,Non native speaker - 41258
* Canadian English - 45640
* England English - 75772
* India and South Asia (India, Pakistan, Sri Lanka) - 79043
* United States English - 249284
```

We also have developed AccentID system for the following languages:

```python
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

# Get started: 

Step 1: Using python 3.9: install python and the requirements

```bash
python -m pip install -r requirements.txt
```

You need to run this to get Pytorch running with CUDA 11.6

Our system is trained on the CommonVoice dataset (11.0 version). Follow the data preparation (`CommonAccent/common_accent_prepare.py`) in `CommonAccent/README.md`
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
