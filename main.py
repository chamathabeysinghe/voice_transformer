import argparse
import speech_recognition as sr
from IPython.utils import io
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import scipy.io.wavfile

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-s', '--source', help='Audio file to change voice', default="samples/test_sample.wav")
parser.add_argument('-t', '--target', help='Sample voice file', default="samples/trump10.wav")
args = vars(parser.parse_args())

print(args['source'])


r = sr.Recognizer()
in_fpath = args['source']
clone_fpath = args['target']
with sr.AudioFile(in_fpath) as source:
    audio_text = r.listen(source)
    text = r.recognize_google(audio_text)
    print('Converting audio transcripts into text ...')
    print(text)



encoder_weights = Path("weights/pretrained/encoder/saved_models/pretrained.pt")
vocoder_weights = Path("weights/pretrained/vocoder/saved_models/pretrained/pretrained.pt")
syn_dir = Path("weights/pretrained/synthesizer/saved_models/logs-pretrained/taco_pretrained")
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)


# in_fpath = Path("trump10.wav")
reprocessed_wav = encoder.preprocess_wav(clone_fpath)
original_wav, sampling_rate = librosa.load(clone_fpath)
preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
embed = encoder.embed_utterance(preprocessed_wav)
with io.capture_output() as captured:
  specs = synthesizer.synthesize_spectrograms([text], [embed])
generated_wav = vocoder.infer_waveform(specs[0])
generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
# display(Audio(generated_wav, rate=synthesizer.sample_rate))
scipy.io.wavfile.write('output_audio.wav', synthesizer.sample_rate, generated_wav)




