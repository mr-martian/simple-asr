#!/usr/bin/env python3

from pyAudioAnalysis import audioBasicIO, audioSegmentation
from xml.etree import ElementTree as ET
import datetime
import itertools
import os

def segment_file(audio_file: str, eaf_file: str):
    frequency, data = audioBasicIO.read_audio_file(audio_file)
    ranges = audioSegmentation.silence_removal(
        data, frequency,
        0.05,  # window size (seconds)
        0.05,  # window step (seconds)
        0.5,   # smooth window (seconds)
        0.2,   # weight / strictness (percentage)
        False, # plot
    )

    document = ET.Element('ANNOTATION_DOCUMENT',
                          AUTHOR='unspecified',
                          DATE=datetime.datetime.now().isoformat(),
                          FORMAT='3.0', VERSION='3.0')
    header = ET.SubElement(document, 'HEADER')
    ET.SubElement(header, 'MEDIA_DESCRIPTOR',
                  MEDIA_URL='file://' + os.path.abspath(audio_file),
                  MIME_TYPE='audio/' + os.path.splitext(audio_file)[1],
                  RELATIVE_MEDIA_URL=audio_file)
    annotation_count = ET.SubElement(header, 'PROPERTY',
                                     NAME='lastUsedAnnotationId')
    annotation_count.text = str(len(ranges))

    time_order = ET.SubElement(document, 'TIME_ORDER')
    time_ids = {}
    for index, time in enumerate(sorted(itertools.chain.from_iterable(ranges))):
        tid = f'ts{index+1}'
        time_ids[time] = tid
        ET.SubElement(time_order, 'TIME_SLOT', TIME_SLOT_ID=tid,
                      TIME_VALUE=str(int(time*1000)))

    tier = ET.SubElement(document, 'TIER', TIER_ID='segments',
                         LINGUISTIC_TYPE_REF='phrase')
    ET.SubElement(document, 'LINGUISTIC_TYPE', LINGUISTIC_TYPE_ID='phrase',
                  TIME_ALIGNABLE='true')
    for index, (start, end) in enumerate(ranges, 1):
        container = ET.SubElement(tier, 'ANNOTATION')
        annotation = ET.SubElement(container, 'ALIGNABLE_ANNOTATION',
                                   ANNOTATION_ID=f'a{index}',
                                   TIME_SLOT_REF1=time_ids[start],
                                   TIME_SLOT_REF2=time_ids[end])
        ET.SubElement(annotation, 'ANNOTATION_VALUE')

    try:
        ET.indent(document)
    except AttributeError:
        # indent() was added in Python 3.9
        pass

    with open(eaf_file, 'wb') as fout:
        fout.write(b'<?xml version="1.0" encoding="UTF-8"?>')
        fout.write(ET.tostring(document, encoding='utf-8'))

def cli():
    import argparse
    parser = argparse.ArgumentParser('Generate a blank EAF from audio')
    parser.add_argument('audio_file', action='store')
    parser.add_argument('eaf_file', action='store')
    args = parser.parse_args()
    segment_file(args.audio_file, args.eaf_file)

if __name__ == '__main__':
    cli()
