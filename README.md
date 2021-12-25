## How to run model?
1. Download/Clone the code repository
2. Download pre-trained weights <a href="https://drive.google.com/u/0/uc?export=download&confirm=I5y5&id=1n1sPXvT34yXFLT47QZA6FIRGrwMeSsZc">here</a>
3. Create a folder named weights and extract content of downloaded folder there.
4. Install required packages using `pip install -r requirements.txt`
5. Run the command `python main.py --source samples/test_sample.wav --target samples/trump10.wav`.
Source is the file to be converted, and target is the sample target voice. 

## Custom training of the voice synthesize model 
Model is trained on LibriSpeech ASR corpus. Download the dataset from <a href="https://www.openslr.org/12/">here</a>

1. Encoder training

```
python encoder_preprocess.py <datasets_root>
python encoder_train.py my_run <datasets_root>/SV2TTS/encoder
```

2. Synthesizer training

```
python synthesizer_preprocess_audio.py <datasets_root>
python synthesizer_preprocess_embeds.py <datasets_root>/SV2TTS/synthesizer
python synthesizer_train.py my_run <datasets_root>/SV2TTS/synthesizer
```

3. Training the vocoder
```
python vocoder_preprocess.py <datasets_root>
python vocoder_train.py my_run <datasets_root>
```