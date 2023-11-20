#!/usr/bin/env python3

from datasets import Dataset
#from ffmpeg import FFmpeg
import numpy as np
import torch
import torchaudio
from transformers import Trainer, TrainingArguments, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import glob
import json
import os.path
import random
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Sequence, Set, Union
import unicodedata
from xml.etree import ElementTree as ET

### Functions

def clean_text_unicode(text: str,
                       charactersToKeep: Optional[Sequence[str]] = None,
                       codeSwitchSymbol: str = '[C]',
                       useCodeSwitchData: bool = True,
                       doubtfulSymbol: str = '[D]',
                       useDoubtfulData: bool = True) -> str:
    '''Strip non-word characters from `text` based on Unicode character classes
    `charactersToKeep`: a list of punctuation characters to retain
    `codeSwitchSymbol`: symbol which indicates that `text` contains codeswitching
    `useCodeSwitchData`: whether codeswitched data should be included in training
    `doubtfulSymbol`: symbol which indicates that `text` is uncertain
    `useDoubtfulData`: where doubtful data should be included in training

    The marker symbols will be removed from `text`. If either marker is present
    and the corresponding flag is False, the function will return an empty
    string.
    '''
    s = text
    if codeSwitchSymbol in s:
        if useCodeSwitchData:
            s = s.replace(codeSwitchSymbol, '')
        else:
            return ''
    if doubtfulSymbol in s:
        if useDoubtfulData:
            s = s.replace(doubtfulSymbol, '')
        else:
            return ''
    ls = []
    cset = set(charactersToKeep or '')
    tset = set(['Ll', 'Lm', 'Lo', 'Lt', 'Lu', 'Mn', 'Nd', 'Nl', 'No'])
    for c in s:
        typ = unicodedata.category(c)
        if c in cset or typ in tset:
            ls.append(c)
        else:
            ls.append(' ')
    return ' '.join(''.join(ls).split())

def load_eaf(path: str, tiernames: List[str]) -> List[Tuple[float, float, str]]:
    '''Read ELAN file and extract annotations in specified tiers
    `path`: path to the ELAN (.eaf) file
    `tiernames`: a list of names of tiers to extract'''
    root = ET.parse(path).getroot()
    times = {}
    for slot in root.iter('TIME_SLOT'):
        times[slot.attrib['TIME_SLOT_ID']] = int(slot.attrib['TIME_VALUE'])/1000
    def get_times(elem):
        if elem.tag == 'ALIGNABLE_ANNOTATION':
            return (times[elem.attrib['TIME_SLOT_REF1']], times[elem.attrib['TIME_SLOT_REF2']])
        else:
            ref_id = elem.attrib['ANNOTATION_REF']
            for ann in root.findall(f".//*[@ANNOTATION_ID='{ref_id}']"):
                return get_times(ann)
    ret = []
    for tiername in tiernames:
        for tier in root.findall(f'.//TIER[@TIER_ID="{tiername}"]'):
            for ann in tier.iter('ANNOTATION'):
                for ann2 in ann:
                    txt = ann2.find('ANNOTATION_VALUE').text
                txt = (txt or '').replace('\n', '').strip()
                if not txt:
                    continue
                start, end = get_times(ann2)
                ret.append((start, end, txt))
    ret.sort()
    return ret

def downsample_audio(path: str, out_dir: str) -> str:
    '''Copy `path` into `out_dir` as a 16kHz .wav file

    Returns the name (not the full path) of the generated file'''
    fname = os.path.basename(path)
    name, ext = os.path.splitext(fname)
    out_name = name + '.wav'
    out_path = os.path.join(out_dir, out_name)
    #f = (
    #    FFmpeg()
    #    .option('y')           # if the output file already exists, overwrite it
    #    .input(path)           # read the input file
    #    .option('ac', '1')     # output as mono (not stereo) sound
    #    .option('ar', '16000') # as 16kHz
    #    .output(out_path)      # write to the output path
    #)
    #f.execute()
    subprocess.run(['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', '16000', out_path], check=True,
                   capture_output=True)
    return out_name

def load_manifest(directory: str) -> Set[str]:
    '''Retrieve the contents of MANIFEST.txt from `directory`

    This is the list of audio files which have already been downsampled
    by previous runs of the preprocessing commands.'''
    path = os.path.join(directory, 'MANIFEST.txt')
    if os.path.exists(path):
        with open(path) as fin:
            return set(fin.read().splitlines())
    else:
        return set()

def add_audio(path: str, out_dir: str,
              overwrite: bool = False,
              manifest: Optional[Set[str]] = None) -> bool:
    '''Add an audio file to the data directory if it does not already exist
    `path`: the path to the audio file
    `out_dir`: the data directory
    `overwrite`: if True, copy the file even if already present (default: False)
    `manifest`: contents of MANIFEST.txt

    If this function is being called in a loop, it can be passed the result
    of calling `load_manifest(out_dir)` to save disk reads.'''
    if manifest is None and not overwrite:
        manifest = load_manifest(out_dir)
    name = os.path.basename(path)
    if overwrite or name not in manifest:
        name = downsample_audio(path, out_dir)
        with open(os.path.join(out_dir, 'MANIFEST.txt'), 'a') as fout:
            fout.write(name + '\n')
        if manifest is not None:
            manifest.add(name)
        return True
    return False

def add_elan_file(audio_path: str, elan_path: str, tiernames: Sequence[str],
                  out_dir: str, overwrite: bool = False,
                  manifest: Optional[Set[str]] = None) -> None:
    isnew = add_audio(audio_path, out_dir, overwrite, manifest)
    if isnew:
        annotations = load_eaf(elan_path, tiernames)
        name = os.path.splitext(os.path.basename(audio_path))[0]
        with open(os.path.join(out_dir, name + '.segments.tsv'), 'w') as fout:
            fout.write(''.join(f'{x[0]}\t{x[1]}\t{x[2]}\n' for x in annotations))

def split_data(directory: str, clean_fn: Callable[[str], str]) -> None:
    manifest = load_manifest(directory)
    annotations = []
    all_chars = set()
    with open(os.path.join(directory, 'clean.tsv'), 'w') as fout:
        fout.write('File Name\tStart Time\tEnd Time\tRaw Text\tCleaned Text\n')
        for audio in sorted(manifest):
            segments = os.path.splitext(audio)[0] + '.segments.tsv'
            with open(os.path.join(directory, segments)) as fin:
                for line in fin:
                    ls = line.split('\t', 2)
                    if len(ls) != 3:
                        continue
                    txt = ls[2].replace('\t', ' ')
                    clean = clean_fn(txt).strip()
                    if not clean:
                        continue
                    all_chars.update(clean)
                    fout.write('\t'.join([audio, ls[0], ls[1], txt, clean])+'\n')
                    annotations.append('\t'.join([audio, ls[0], ls[1], clean]))
    random.shuffle(annotations)
    n1 = round(len(annotations)*0.8)
    n2 = round(len(annotations)*0.9)
    sections = {
        'train': annotations[:n1],
        'dev': annotations[n1:n2],
        'test': annotations[n2:],
    }
    for name, lines in sections.items():
        with open(os.path.join(directory, name+'.tsv'), 'w') as fout:
            fout.write('audio\tstart\tend\ttext\n')
            fout.write('\n'.join(lines) + '\n')
    char_dict = {c:i for i,c in enumerate(sorted(all_chars))}
    char_dict['|'] = char_dict[' ']
    del char_dict[' ']
    char_dict['[UNK]'] = len(char_dict)
    char_dict['[PAD]'] = len(char_dict)
    with open(os.path.join(directory, 'vocab.json'), 'w') as fout:
        json.dump(char_dict, fout)

def make_processor(data_dir: str, model_dir: Optional[str] = None) -> Wav2Vec2Processor:
    vocab = os.path.join(data_dir, 'vocab.json')
    tokenizer = Wav2Vec2CTCTokenizer(vocab, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    if model_dir is not None:
        processor.save_pretrained(model_dir)
    return processor

def load_samples(path: str, processor: Wav2Vec2Processor) -> Any: # TODO
    sampling_rate = 16000
    dirname = os.path.dirname(path)
    def load_audio(entry):
        nonlocal sampling_rate, dirname
        start = int(float(entry['start'])*sampling_rate)
        end = int(float(entry['end'])*sampling_rate)
        speech, _ = torchaudio.load(os.path.join(dirname, entry['audio']),
                                    frame_offset=start,
                                    num_frames=(end-start))
        entry['speech'] = speech[0].numpy()
        return entry
    def pad_audio(batch):
        nonlocal sampling_rate, processor
        batch['input_values'] = processor(batch['speech'], sampling_rate=sampling_rate).input_values
        with processor.as_target_processor():
            batch['labels'] = processor(batch['text']).input_ids
        return batch
    data = Dataset.from_csv(path, sep='\t')
    # There have to be 2 separate map() steps because one is batched and the other isn't
    return data.map(load_audio).map(pad_audio, batch_size=8, num_proc=4, batched=True)

# originally from https://github.com/huggingface/transformers/blob/9a06b6b11bdfc42eea08fa91d0c737d1863c99e3/examples/research_projects/wav2vec2/run_asr.py#L81
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def metric_computer(metrics, processor):
    def compute_metrics(prediction):
        pred_logits = prediction.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        prediction.label_ids[prediction.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        ret = {}
        for k, v in metrics.items():
            ret[k] = v.compute(predictions=pred_str, references=label_str)
        return ret
    return compute_metrics

def compute_wer(processor):
    import evaluate
    return metric_computer({'wer': evaluate.load('wer')}, processor)

def train_on_data(processor, out_dir, train, dev,
                  epochs: int = 100):
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_extractor()
    training_args = TrainingArguments(
        output_dir=out_dir,
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=epochs,
        fp16=True,
        save_steps=400,
        eval_steps=100,
        logging_steps=50,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=10,
    )
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_wer(processor),
        train_dataset=train,
        eval_dataset=dev,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()

def list_checkpoints(model_dir: str):
    return glob.glob('checkpoint-*', root_dir=model_dir)

def load_processor(model_dir: str):
    return Wav2Vec2Processor.from_pretrained(model_dir)

def load_checkpoint(model_dir: str, checkpoint: str):
    return Wav2Vec2ForCTC.from_pretrained(os.path.join(model_dir, checkpoint)).to('cuda')

def predict_tensor(tensor, model, processor):
    input_dict = processor(tensor, return_tensors='pt', padding=True)
    logits = model(input_dict.input_values.to('cuda')).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    return processor.decode(pred_ids)

def predict_test_set(data, model, processor):
    def pred(sample):
        sample['prediction'] = predict_tensor(sample['input_values'], model, processor)
        return sample
    return data.map(pred)

### CLI

def cli_elan():
    parser = argparse.ArgumentParser('Preprocess ELAN files for ASR training')
    parser.add_argument('audio', action='store')
    parser.add_argument('eaf', action='store')
    parser.add_argument('data_dir', action='store')
    parser.add_argument('tiers', nargs='+')
    parser.add_argument('--overwrite', '-o', action='store_true')
    args = parser.parse_args()
    add_elan_file(args.audio, args.eaf, args.tiers, args.data_dir,
                  overwrite=args.overwrite)

def cli_cv():
    pass

def cli_train():
    parser = argparse.ArgumentParser('Train an ASR model')
    parser.add_argument('data_dir', action='store')
    args = parser.parse_args()
    proc = make_processor(args.data_dir)
    train = load_samples(os.path.join(args.data_dir, 'train.tsv'), proc)
    dev = load_samples(os.path.join(args.data_dir, 'dev.tsv'), proc)
    test = load_samples(os.path.join(args.data_dir, 'test.tsv'), proc)
    train_on_data(proc, args.data_dir, train, dev)

def cli_split():
    parser = argparse.ArgumentParser('Split data into train, dev, and test sections')
    parser.add_argument('data_dir', action='store')
    parser.add_argument('--keep', '-k', action='store')
    args = parser.parse_args()
    split_data(args.data_dir,
               lambda s: clean_text_unicode(s, charactersToKeep=list(args.keep or '')).lower())

def cli_predict():
    pass
