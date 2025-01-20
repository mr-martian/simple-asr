#!/usr/bin/env python3

from datasets import Dataset
import evaluate
import jiwer
import numpy as np
import torch
import torchaudio
from transformers import EvalPrediction, Trainer, TrainingArguments, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC

import argparse
import csv
from dataclasses import dataclass
import glob
import json
import os.path
import pathlib
import random
import statistics
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence, Set, Union
import unicodedata
from xml.etree import ElementTree as ET

### Functions

def clean_text_unicode(text: str,
                       charactersToKeep: Optional[Sequence[str]] = None,
                       codeSwitchSymbol: str = '[C]',
                       useCodeSwitchData: bool = True,
                       doubtfulSymbol: str = '[D]',
                       useDoubtfulData: bool = True) -> str:
    """Strip non-word characters based on Unicode character classes.

    :param text: The string to be cleaned
    :type text: str
    :param charactersToKeep: A list of punctuation characters to retain
    :type charactersToKeep: list, optional
    :param codeSwitchSymbol: A symbol which indicates that the contents of
        ``text`` include codeswitching. The symbol will be removed from ``text``,
        defaults to `'[C]'`
    :type codeSwitchSymbol: str, optional
    :param useCodeSwitchData: If `False` and ``codeSwitchSymbole`` is present
        in ``text``, return an empty string, defaults to `True`
    :type useCodeSwitchData: bool, optional
    :param doubtfulSymbol: A symbol that incdicates that the contents of
        ``text`` are uncertain. The symbol will be removed from ``text``,
        defaults to `'[D]'`
    :type doubtfulSymbol: str, optional
    :param useDoubtfulData: where doubtful data should be included in training
    :type useDoubtfulData: bool, optional
    :return: The cleaned string, or an empty string if codeswitched or
        uncertain annotations have been excluded
    :rtype: str

    The marker symbols will be removed from `text`. If either marker is present
    and the corresponding flag is False, the function will return an empty
    string.
    """
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
    """Read an ELAN file and extract annotations in specified tiers

    :param path: The path to the ELAN (.eaf) file
    :type path: str
    :param tiernames: A list of names of tiers to extract
    :type tiernames: list
    :return: A list of non-empty annotations as tuples of `(start, end, text)`
    :rtype: list
    """
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

def downsample_audio(path: str, data_dir: str) -> str:
    """Copy an audio file to a specified directory as a 16kHz .wav file

    :param path: The path to the original file
    :type path: str
    :param data_dir: The directory for preprocessed training data
    :type data_dir: str
    :return: The name (not the full path) of the generated file
    :rtype: str
    """
    fname = os.path.basename(path)
    name, ext = os.path.splitext(fname)
    out_name = name + '.wav'
    out_path = os.path.join(data_dir, out_name)
    subprocess.run(['ffmpeg', '-y',      # overwrite
                    '-i', path,          # read input file
                    '-ac', '1',          # output as mono sound (not stereo)
                    '-ar', '16000',      # output as 16kHz
                    out_path],           # write to output file
                   check=True,           # propogate errors to caller
                   capture_output=True)  # don't show debug info
    return out_name

def load_manifest(directory: str) -> Set[str]:
    """Retrieve the contents of the manifest file from a directory.

    :param directory: The directory to read from
    :type directory: str
    :return: The names of the audio files listed in the manifest
    :rtype: set

    Preprocessing an audio file via :func:`add_audio` causes it to be
    listed in ``MANIFEST.txt`` (one filename per line, unsorted).
    """
    path = os.path.join(directory, 'MANIFEST.txt')
    if os.path.exists(path):
        with open(path) as fin:
            return set(fin.read().splitlines())
    else:
        return set()

def add_audio(path: str, data_dir: str,
              overwrite: bool = False,
              manifest: Optional[Set[str]] = None) -> bool:
    """Add an audio file to the data directory if it does not already exist.

    :param path: The path to the audio file
    :type path: str
    :param data_dir: The directory for preprocessed training data
    :type data_dir: str
    :param overwrite: If ``True``, do not check whether the file has already
        been copied to `data_dir`, defaults to ``False``
    :type overwrite: bool, optional
    :param manifest: The contents of the manifest file
        (see :func:`load_manifest`)
    :type manifest: set, optional

    If this function is being called in a loop, it can be passed the result
    of calling ``load_manifest(data_dir)`` to save disk reads.
    """
    pathlib.Path(data_dir).mkdir(exist_ok=True)
    if manifest is None and not overwrite:
        manifest = load_manifest(data_dir)
    name = os.path.basename(path)
    if overwrite or name not in manifest:
        name = downsample_audio(path, data_dir)
        with open(os.path.join(data_dir, 'MANIFEST.txt'), 'a') as fout:
            fout.write(name + '\n')
        if manifest is not None:
            manifest.add(name)
        return True
    return False

def add_elan_file(audio_path: str, elan_path: str, tiernames: Sequence[str],
                  data_dir: str, overwrite: bool = False,
                  manifest: Optional[Set[str]] = None) -> None:
    """Convert an ELAN file to a segments file and add the matching audio file.

    :param audio_path: The path to the audio file
    :type audio_path: str
    :param elan_path: The path to the ELAN (.eaf) file
    :type elan_path: str
    :param tiernames: The names of the tiers to extract into the segments file
    :type tiernames: list
    :param data_dir: The directory for preprocessed training data
    :type data_dir: str
    :param overwrite: Whether to overwrite output files if they already exist,
        defaults to ``False``
    :type overwrite: bool, optional
    :param manifest: The contents of the manifest file
        (see :func:`load_manifest`)
    :type manifest: set, optional
    """
    isnew = add_audio(audio_path, data_dir, overwrite, manifest)
    if isnew:
        annotations = load_eaf(elan_path, tiernames)
        name = os.path.splitext(os.path.basename(audio_path))[0]
        with open(os.path.join(data_dir, name + '.segments.tsv'), 'w') as fout:
            fout.write(''.join(f'{x[0]}\t{x[1]}\t{x[2]}\n' for x in annotations))

def add_common_voice_files(cv_dir: str, data_dir: str, overwrite: bool = False,
                           use_other: bool = False) -> None:
    """Add audio files from a Common Voice download.

    :param cv_dir: The directory of the Common Voice download
    :type cv_dir: str
    :param data_dir: The directory for preprocessed training data
    :type data_dir: str
    :param overwrite: Whether to overwrite output files if they already exist,
        defaults to ``False``
    :type overwrite: bool, optional
    :param use_other: Whether to include non-validated audio clips,
        defaults to ``False``
    :type use_other: bool, optional

    This function will read the contents of ``validated.tsv`` in ``cv_dir``,
    and will add ``other.tsv`` if ``use_other`` is ``True``.
    """
    times = {}
    with open(os.path.join(cv_dir, 'times.txt')) as fin:
        for line in fin:
            if not line.strip():
                continue
            fname = line.split('`')[1].split('/')[-1]
            tm = line.split('=')[-1].strip()
            times[fname] = int(tm) / 1000
    tsv_files_to_read = [os.path.join(cv_dir, 'validated.tsv')]
    if use_other:
        tsv_files_to_read.append(os.path.join(cv_dir, 'other.tsv'))
    for tsv_fname in tsv_files_to_read:
        with open(tsv_fname) as fin:
            for i, line in enumerate(fin):
                if i == 0 or not line.strip():
                    continue
                ls = line.split('\t')
                audio = ls[1]
                segments = os.path.splitext(audio)[0] + '.segments.tsv'
                sentence = ls[2]
                add_audio(os.path.join(cv_dir, 'clips', audio), data_dir,
                          overwrite=overwrite)
                with open(os.path.join(data_dir, segments), 'w') as fout:
                    fout.write(f'0.0\t{times[audio]}\t{sentence}\n')

def split_data(directory: str, clean_fn: Callable[[str], str]) -> None:
    """Split data into training, development, and testing sections.

    :param directory: The directory to read from
    :type directory: str
    :param clean_fn: A callback to clean the text

    Load the manifest (see :func:`load_manifest`) and read the segments
    file for each listed audio file. Shuffle this list and assign 80% to
    training, 10% to development, and 10% to testing. Apply the cleaning
    function and write as 3 .tsv files: ``train.tsv``, ``dev.tsv``, and
    ``test.tsv``. Additionally, write the alphabet of the cleaned text
    to ``vocab.json``.
    """
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

def make_processor(data_dir: str, model_dir: str) -> Wav2Vec2Processor:
    """Create a new data processor.

    :param data_dir: The directory for preprocessed training data
    :type data_dir: str
    :param model_dir: The directory for saved model files
    :type model_dir: str
    :return: A processor object
    :rtype: :class:`transformers.Wav2Vec2Processor`

    This function creates a tokenizer using the ``vocab.json`` file created
    by :func:`split_data` and an audio feature extractor and saves the result
    to ``model_dir``.
    """
    vocab = os.path.join(data_dir, 'vocab.json')
    tokenizer = Wav2Vec2CTCTokenizer(vocab, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(model_dir)
    return processor

def load_samples(path: str, processor: Wav2Vec2Processor) -> Dataset:
    """Load a data-split .tsv file as a `Dataset`.

    :param path: The path to the file
    :type path: str
    :param processor: A processor object to encode the audio with
    :type processor: :class:`transformers.Wav2Vec2Processor`
    :return: A Dataset containing the audio samples and transcriptions
    :rtype: :class:`datasets.Dataset`

    The returned dataset has the following keys:

    * `audio` (`str`): The filename of the audio sample
    * `start` (`float`): The start time within the file in seconds
    * `end` (`float`): The end time within the file in seconds
    * `text` (`str`): The text of the sample
    * `speech` (:class:`numpy.ndarray`): The audio signal
    * `input_values` (:class:`torch.Tensor`): The input features to the model
    * `labels` (:class:`torch.Tensor`): The expected model output values
    """
    sampling_rate = 16000
    dirname = os.path.dirname(path)
    def load(entry):
        nonlocal sampling_rate, dirname
        start = int(float(entry['start'])*sampling_rate)
        end = int(float(entry['end'])*sampling_rate)
        speech, _ = torchaudio.load(os.path.join(dirname, entry['audio']),
                                    frame_offset=start,
                                    num_frames=(end-start))
        entry['speech'] = speech[0].numpy()
        return entry
    def pad(batch):
        nonlocal sampling_rate, processor
        ret = processor(batch['speech'], sampling_rate=sampling_rate,
                        text=batch['text'])
        batch['input_values'] = ret.input_values
        batch['labels'] = ret.labels
        return batch
    data = Dataset.from_csv(path, sep='\t')
    # There have to be 2 separate map() steps because one is batched
    # and the other isn't
    return data.map(load).map(pad, batch_size=8, num_proc=4, batched=True)

# originally from https://github.com/huggingface/transformers/blob/9a06b6b11bdfc42eea08fa91d0c737d1863c99e3/examples/research_projects/wav2vec2/run_asr.py#L81
@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that will dynamically pad the inputs received.

    :ivar processor: The processor used for proccessing the data.
    :ivar padding: Select a strategy to pad the returned sequences (according to
        the model's padding side and padding index) among:

        * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
        * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not provided.
        * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

    :ivar max_length: Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
    :ivar max_length_labels: Maximum length of the ``labels`` returned list and optionally padding length (see above).
    :ivar pad_to_multiple_of: If set will pad the sequence to a multiple of the
        provided value. This is especially useful to enable the use of Tensor
        Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and
        # need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def metric_computer(metrics: Dict[str, evaluate.EvaluationModule],
                    processor: Wav2Vec2Processor) -> Callable[[EvalPrediction],
                                                              Dict[str, float]]:
    """Generate a callback for computing metrics during training.

    :param metrics: A dictionary mapping names to evaluation metrics loaded
        with :func:`evaluate.load`
    :type metrics: dict
    :param processor: The feature processor
    :type processor: :class:`transformers.Wav2Vec2Processor`
    :return: A callback
    :rtype: function
    """
    def compute_metrics(prediction):
        pred_logits = prediction.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        prediction.label_ids[prediction.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(prediction.label_ids, group_tokens=False)
        ret = {}
        for k, v in metrics.items():
            ret[k] = v.compute(predictions=pred_str, references=label_str)
        return ret
    return compute_metrics

def compute_wer(processor: Wav2Vec2Processor) -> Callable[[EvalPrediction],
                                                          Dict[str, float]]:
    """Generate a callback for computing Word-Error Rate during training.

    :param processor: The feature processor
    :type processor: :class:`transformers.Wav2Vec2Processor`
    :return: A callback
    :rtype: function
    """
    return metric_computer({'wer': evaluate.load('wer')}, processor)

def train_on_data(processor: Wav2Vec2Processor, model_dir: str, train: Dataset,
                  dev: Dataset, epochs: int = 100, fp16: bool = False) -> None:
    """Train a model on a particular set of data.

    :param processor: The feature processor
    :type processor: :class:`transformers.Wav2Vec2Processor`
    :param model_dir: The directory where the model will be saved
    :type model_dir: str
    :param train: The training data (see :func:`load_samples`)
    :type train: :class:`datasets.Dataset`
    :param dev: The development (validation) data (see :func:`load_samples`)
    :type dev: :class:`datasets.Dataset`
    :param epochs: The number of epochs to train, defaults to ``100``
    :type epochs: int, optional
    :param fp16: Whether to use half-precision floating point for training,
        defaults to ``False``
    :type fp16: bool, optional
    """
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.freeze_feature_encoder()
    training_args = TrainingArguments(
        output_dir=model_dir,
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=epochs,
        fp16=fp16,
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

def train(data_dir: str, model_dir: str, **kwargs) -> None:
    """Train a model.

    :param data_dir: The directory containing the preprocessed data
    :type data_dir: str
    :param model_dir: The directory where the model will be saved
    :type model_dir: str

    The keyword arguments for this function are the same as for
    :func:`train_on_data`.
    """
    processor = make_processor(data_dir, model_dir)
    train = load_samples(os.path.join(data_dir, 'train.tsv'), processor)
    dev = load_samples(os.path.join(data_dir, 'dev.tsv'), processor)
    train_on_data(processor, model_dir, train, dev, **kwargs)

def list_checkpoints(model_dir: str) -> List[str]:
    """Return a list of checkpoint names found in a directory.

    :param model_dir: The directory where a model was saved
    :type model_dir: str
    :return: The list of checkpoints
    :rtype: list

    Checkpoint names are of the form ``'checkpoint-####'``, where ``'####'`` is the step number.
    """
    return sorted(glob.glob('checkpoint-*', root_dir=model_dir),
                  key=lambda x: int(x[11:]) if x[11:].isdigit() else 0)

def load_processor(model_dir: str) -> Wav2Vec2Processor:
    """Load a saved processor.

    :param model_dir: The directory where a model was saved
    :type model_dir: str
    :return: The processor
    :rtype: :class:`transformers.Wav2Vec2Processor`
    """
    return Wav2Vec2Processor.from_pretrained(model_dir)

def load_checkpoint(model_dir: str, checkpoint: str) -> Wav2Vec2ForCTC:
    """Load a model checkpoint.

    :param model_dir: The directory where a model was saved
    :type model_dir: str
    :param checkpoint: The name of the checkpoint
    :type checkpoint: str
    :return: The model
    :rtype: :class:`transformers.Wav2Vec2ForCTC`
    """
    pth = os.path.join(model_dir, checkpoint)
    return Wav2Vec2ForCTC.from_pretrained(pth).to('cuda')

def predict_tensor(tensor: torch.Tensor, model: Wav2Vec2ForCTC,
                   processor: Wav2Vec2Processor) -> str:
    """Pass a feature tensor to a trained model and return the predicted string.

    :param tensor: The input audio clip
    :type tensor: :class:`torch.Tensor`
    :param model: The trained model
    :type model: :class:`transformers.Wav2Vec2ForCTC`
    :param processor: The feature processor
    :type processor: :class:`transformers.Wav2Vec2Processor`
    :return: The predicted string
    :rtype: str
    """
    input_dict = processor(tensor, return_tensors='pt', padding=True,
                           sampling_rate=16000)
    logits = model(input_dict.input_values.to('cuda')).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    return processor.decode(pred_ids)

def predict_test_set(data: Dataset, model: Wav2Vec2ForCTC,
                     processor: Wav2Vec2Processor) -> Dataset:
    """Predict labels for all samples in a dataset.

    :param data: The input dataset (see :func:`load_samples`)
    :type data: :class:`datasets.Dataset`
    :param model: The trained model
    :type model: :class:`transformers.Wav2Vec2ForCTC`
    :param processor: The feature processor
    :type processor: :class:`transformers.Wav2Vec2Processor`
    :return: The input dataset with the added key `prediction` (`str`)
        containing the predicted text
    :rtype: :class:`datasets.Dataset`
    """
    def pred(sample):
        sample['prediction'] = predict_tensor(sample['input_values'], model, processor)
        return sample
    return data.map(pred)

def evaluate_test_set(data: Dataset) -> Dataset:
    """Calculate the error rate for a dataset with predictions.

    :param data: The input dataset (output of :func:`predict_test_set`)
    :type data: :class:`datasets.Dataset`
    :return: The input datset with the added keys `wer` (Word-Error Rate)
        and `cer` (Character-Error Rate)
    :rtype: :class:`datasets.Dataset`
    """
    def ev(sample):
        sample['wer'] = jiwer.wer(sample['text'].lower(), sample['prediction'])
        sample['cer'] = jiwer.cer(sample['text'].lower(), sample['prediction'])
        return sample
    return data.map(ev)

def evaluate_checkpoint(data: Dataset, model_dir: str, checkpoint: str,
                        processor: Optional[Wav2Vec2Processor] = None,
                        out_file: Optional[str] = None) -> Tuple[float, float]:
    """Evaluate a saved checkpoint on a particular set of data.

    :param data: The input dataset (see :func:`load_samples`)
    :type data: :class:`datasets.Dataset`
    :param model_dir: The directory where a model was saved
    :type model_dir: str
    :param checkpoint: The name of the checkpoint
    :type checkpoint: str
    :param processor: The feature processor, or ``None`` to load it from
        ``model_dir``, defaults to ``None``
    :type processor: :class:`transformers.Wav2Vec2Processor`, optional
    :param out_file: A file to write individual predictions to, for further
        analysis, defaults to ``None``
    :type out_file: str, optional
    :return: The median Character-Error Rate (CER) and Word-Error Rate (WER)
        of samples in the dataset as floats between 0 and 1
    :rtype: tuple
    """
    proc = processor if processor is not None else load_processor(model_dir)
    model = load_checkpoint(model_dir, checkpoint)
    pred_data = evaluate_test_set(predict_test_set(data, model, proc))
    cer = statistics.median([s['cer'] for s in pred_data])
    wer = statistics.median([s['wer'] for s in pred_data])
    if out_file:
        lns = ['file\texpected\toutput\tWER\tCER\n']
        for s in pred_data:
            lns.append(f"{s['audio']}\t{s['text']}\t{s['prediction']}\t{round(s['wer'],2)}\t{round(s['cer'],2)}\n")
        with open(out_file, 'w') as fout:
            fout.write(''.join(lns))
    return cer, wer

def evaluate_all_checkpoints(
        data_dir: str, model_dir: str,
        log_dir: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
    """Evaluate every saved checkpoint from a training run.

    :param data_dir: The directory for preprocessed training data
    :type data_dir: str
    :param model_dir: The directory for saved model files
    :type model_dir: str
    :param log_dir: A directory to save detailed evaluation files to,
        defaults to ``None``
    :type log_dir: str, optional
    :return: A dictionary mapping checkpoint names to tuples of CER and WER
        (as returned by :func:`evaluate_checkpoint`)
    :rtype: dict

    In addition to the files produced by :func:`evaluate_checkpoint`,
    this function produces a file named ``median.tsv`` in ``log_dir`` which
    contains the numbers in the returned dictionary.
    """
    proc = load_processor(model_dir)
    checkpoints = list_checkpoints(model_dir)
    data = load_samples(os.path.join(data_dir, 'test.tsv'), proc)
    scores = {}
    for chk in checkpoints:
        out_file = None
        if log_dir is not None:
            out_file = os.path.join(log_dir, f'{chk}.log.tsv')
        scores[chk] = evaluate_checkpoint(data, model_dir, chk, proc, out_file)
    if log_dir is not None:
        with open(os.path.join(log_dir, 'median.tsv'), 'w') as fout:
            fout.write('checkpoint\tCER\tWER\n')
            for chk in sorted(scores, key=lambda c: int(c.split('-')[1])):
                cer, wer = scores[chk]
                fout.write(f'{chk}\t{cer}\t{wer}\n')
    return scores

def predict_audio_file(infile: str, model: Wav2Vec2ForCTC,
                       processor: Wav2Vec2Processor) -> List[Tuple[float, float, str]]:
    """Transcribe audio segments from a file.

    :param infile: The input audio file
    :type infile: str
    :param model: The trained model
    :type model: :class:`transformers.Wav2Vec2ForCTC`
    :param processor: The feature processor
    :type processor: :class:`transformers.Wav2Vec2Processor`
    :return: Predicted strings with start and end time stamps
    :rtype: list
    """
    with tempfile.TemporaryDirectory() as tmp:
        fname = downsample_audio(infile, tmp)
        speech, _ = torchaudio.load(os.path.join(tmp, fname))

        # TODO: this just splits into 10-second chunks
        # is it worth doing something more intelligent?
        sampling_rate = 16000
        step = sampling_rate * 10
        ret = []
        for i in range(0, len(speech), step):
            ret.append((float(i) / sampling_rate,
                        float(i+step) / sampling_rate,
                        predict_tensor(speech[i:i+step], model, processor)))
        return ret

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
    parser = argparse.ArgumentParser('Preprocess a Mozilla Common Voice directory for ASR training')
    parser.add_argument('cv_dir', action='store')
    parser.add_argument('data_dir', action='store')
    parser.add_argument('--overwrite', '-o', action='store_true')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Include clips listed in other.tsv in addition to validated.tsv')
    args = parser.parse_args()
    add_common_voice_files(args.cv_dir, args.data_dir,
                           overwrite=args.overwrite, use_other=args.all)

def cli_split():
    parser = argparse.ArgumentParser('Split data into train, dev, and test sections')
    parser.add_argument('data_dir', action='store')
    parser.add_argument('--keep', '-k', action='store')
    args = parser.parse_args()
    split_data(args.data_dir,
               lambda s: clean_text_unicode(s, charactersToKeep=list(args.keep or '')).lower())

def cli_train():
    parser = argparse.ArgumentParser('Train an ASR model')
    parser.add_argument('data_dir', action='store')
    parser.add_argument('model_dir', action='store')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    train(args.data_dir, args.model_dir, epochs=args.epochs, fp16=args.fp16)

def cli_eval():
    parser = argparse.ArgumentParser('Evaluate an ASR model')
    parser.add_argument('data_dir', action='store')
    parser.add_argument('model_dir', action='store')
    args = parser.parse_args()
    evaluate_all_checkpoints(args.data_dir, args.model_dir, args.model_dir)
    print(f'Wrote evaluation logs to {args.model_dir}.')

def cli_predict():
    parser = argparse.ArgumentParser('Use an ASR model to transcribe files')
    parser.add_argument('model_dir', action='store')
    parser.add_argument('audio_file', action='store')
    parser.add_argument('output', action='store')
    parser.add_argument('--checkpoint', action='store')
    args = parser.parse_args()
    processor = load_processor(args.model_dir)
    cp = args.checkpoint
    if not cp:
        cp = list_checkpoints(args.model_dir)[-1]
    model = load_checkpoint(args.model_dir, cp)
    with open(args.output, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['start', 'end', 'text'])
        writer.writerows(predict_audio_file(args.audio_file, model,
                                            processor))
    print(f'Wrote transcript to {args.audio_file}.')
