# simple-asr
Wrapper module around wav2vec2 designed for ease of use

# Installation

Ensure that [`ffmpeg`](https://ffmpeg.org/download.html) is installed.

Then do `pip install simple-asr`

# Collab Notebook
```
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)
!pip install simple-asr
!simple-asr-elan /content/drive/MyDrive/path/to/audio.wav /content/drive/MyDrive/path/to/elan.eaf /content/data default
!simple-asr-split /content/data
!simple-asr-train /content/data /content/model -e 140
!simple-asr-evaluate /content/data /content/model
```

This will result in there being model checkpoints in `/content/model` which can then be copied to your drive.
