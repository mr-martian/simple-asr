# easy-asr
Wrapper module around wav2vec2 designed for ease of use

# Installation

Ensure that [`ffmpeg`](https://ffmpeg.org/download.html) is installed.

Then clone this repo and run `pip3 install .` from inside this directory.

# Collab Notebook

```
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)
!git clone https://github.com/mr-martian/easy-asr.git
!pip install /content/easy-asr
!easy-asr-elan /content/drive/MyDrive/path/to/audio.wav /content/drive/MyDrive/path/to/elan.eaf /content/data default
!easy-asr-split /content/data
!easy-asr-train /content/data /content/model -e 140
!easy-asr-evaluate /content/data /content/model
```

This will result in there being model checkpoints in `/content/model` which can then be copied to your drive.
