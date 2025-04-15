import argparse
from collections import defaultdict
import json
import os.path
import simple_asr
import tgt
import torch
import torchaudio
from torchaudio.functional import forced_align, merge_tokens

def align(model, processor, audio, vocab, clean_transcript: str):
    with torch.inference_mode():
        emission = model(audio).logits
    num_frames = emission.size(1)
    tokens = [vocab.get(c, 0) for c in clean_transcript]
    tokens = [t for t in tokens if t != 0]
    tokens = torch.tensor([tokens], dtype=torch.int32, device='cuda')
    alignments, scores = forced_align(emission, tokens, blank=0)
    return merge_tokens(alignments[0], scores[0]), num_frames

def to_textgrid(letter_tier, word_tier, offset: float, ratio: float,
                spans, transcript, vocab):
    word_start = None
    word_end = None
    word = ''
    i = 0
    txt = transcript + ' '
    for span in spans:
        while txt[i] not in vocab:
            if txt[i] == ' ':
                if word:
                    word_tier.add_interval(tgt.core.Interval(
                        word_start, word_end, word))
                word_start = None
                word_end = None
                word = ''
            else:
                word += txt[i]
            i += 1
        start = (span.start * ratio / simple_asr.SAMPLING_RATE) + offset
        end = (span.end * ratio / simple_asr.SAMPLING_RATE) + offset
        letter_tier.add_interval(tgt.core.Interval(
            start, end, txt[i]))
        if word_start is None:
            word_start = start
        word_end = end
        word += txt[i]
        i += 1
    if word:
        word_tier.add_interval(tgt.core.Interval(
            word_start, word_end, word))

def align_file(path: str, model, processor, vocab, textgrid_path: str,
               clean_fn=simple_asr.clean_text_unicode):
    sentences = tgt.core.IntervalTier(name='Sentence')
    words = tgt.core.IntervalTier(name='Word')
    letters = tgt.core.IntervalTier(name='Letter')
    segments = os.path.splitext(path)[0] + '.segments.tsv'
    with open(segments) as fin:
        for line in fin:
            ls = line.split('\t', 2)
            if len(ls) != 3:
                continue
            start = float(ls[0])
            end = float(ls[1])
            txt = clean_fn(ls[2].replace('\t', ' ')).strip().lower()
            if not txt:
                continue
            speech, _ = torchaudio.load(
                path, frame_offset=int(start * simple_asr.SAMPLING_RATE),
                num_frames=int((end - start) * simple_asr.SAMPLING_RATE))
            spans, num_frames = align(model, processor, speech.to('cuda'),
                                      vocab, txt)
            # TODO: why do we need this?
            ratio = speech.size(1) / num_frames
            to_textgrid(letters, words, start, ratio, spans, txt, vocab)
            sentences.add_interval(tgt.core.Interval(start, end, txt))
    grid = tgt.core.TextGrid()
    grid.add_tier(sentences)
    grid.add_tier(words)
    grid.add_tier(letters)
    tgt.io.write_to_file(grid, textgrid_path)

def align_all_training_data(data_dir: str, model_dir: str,
                            textgrid_dir: str, checkpoint=None):
    processor = simple_asr.load_processor(model_dir)
    model = simple_asr.load_checkpoint(model_dir, checkpoint)
    manifest = sorted(simple_asr.load_manifest(data_dir))
    with open(os.path.join(model_dir, 'vocab.json')) as fin:
        vocab = json.load(fin)
    for audio in manifest:
        tg_path = os.path.join(textgrid_dir,
                               os.path.splitext(audio)[0]+'.TextGrid')
        align_file(os.path.join(data_dir, audio), model, processor,
                   vocab, tg_path)

def cli_align_training_data():
    parser = argparse.ArgumentParser('Train an ASR model')
    parser.add_argument('data_dir', action='store')
    parser.add_argument('model_dir', action='store')
    parser.add_argument('textgrid_dir', action='store')
    parser.add_argument('--checkpoint', action='store')
    args = parser.parse_args()
    align_all_training_data(args.data_dir, args.model_dir,
                            args.textgrid_dir, args.checkpoint)
