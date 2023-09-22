"""
Use this script to create JSON-Line description files that can be used to
train deep-speech models through this library.
This works with data directories that are organized like LibriSpeech:
data_directory/group/speaker/[file_id1.wav, file_id2.wav, ...,
                              speaker.trans.txt]
Where speaker.trans.txt has in each line, file_id transcription
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import wave


def main():
    labels = []
    durations = []
    keys = []
    data_directory = '/home/wangye/wangye/data/LibriSpeech/train-other-500'

    for group in os.listdir(data_directory):

        if group.startswith('.'):
            continue

        speaker_path = os.path.join(data_directory, group)
        print(speaker_path)
        for speaker in os.listdir(speaker_path):
            if speaker.startswith('.'):
                continue
            labels_file = os.path.join(speaker_path, speaker,
                                       '{}-{}.trans.txt'
                                       .format(group, speaker))
            for line in open(labels_file):
                split = line.strip().split()
                file_id = split[0]
                label = ' '.join(split[1:]).lower()
                audio_file = os.path.join(speaker_path, speaker,
                                          file_id) + '.flac'
                import librosa
                time = librosa.get_duration(filename=audio_file)
                duration = float(time)
                keys.append(audio_file)
                durations.append(duration)
                labels.append(label)

    output_file = '/home/wangye/wangye/data/LibriSpeech/train-other-500.json'

    with open(output_file, 'w') as out_file:
        for i in range(len(keys)):
            line = json.dumps({'key': keys[i], 'duration': durations[i],
                              'sentence': labels[i]})
            out_file.write(line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('data_directory', type=str,
    #                     help='Path to data directory')
    # parser.add_argument('output_file', type=str,
    #                     help='Path to output file')
    # args = parser.parse_args()
    main()