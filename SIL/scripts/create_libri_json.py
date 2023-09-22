from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import wave
from bert_embedding import BertEmbedding
import numpy as np

def main(data_directory, output_file):
    bert_embedding = BertEmbedding(max_seq_length=50)
    vocab = bert_embedding.vocab
    labels = []
    durations = []
    keys = []
    words = []
    timestamps = []
    file_ids = []
    bert_feats = []

    unaligned_list = ['1992-141719-0042','2289-152257-0025','5456-62014-0010','7067-76048-0021']
    data_directory2 = '/home/wangye/wangye/data/LibriSpeech-Alignments/LibriSpeech/train-clean-100'
    for group in os.listdir(data_directory):
        speaker_path = os.path.join(data_directory, group)
        for speaker in os.listdir(speaker_path):
            print(speaker)
            labels_file = os.path.join(speaker_path, speaker,
                                       '{}-{}.trans.txt'
                                       .format(group, speaker))
            for line in open(labels_file):
                split = line.strip().split()
                file_id = split[0]
                label = ' '.join(split[1:]).lower()
                if file_id in unaligned_list:
                    continue
                audio_file = os.path.join(speaker_path, speaker,
                                          file_id) + '.flac'
                import librosa
                time = librosa.get_duration(filename=audio_file)
                #print(time)
                # audio = wave.open(audio_file)
                duration = float(time)
                # audio.close()
                file_ids.append(file_id)
                keys.append(audio_file)
                durations.append(duration)
                labels.append(label)

        speaker_path2 = os.path.join(data_directory2, group)
        for speaker in os.listdir(speaker_path2):
            labels_file = os.path.join(speaker_path2, speaker,
                                       '{}-{}.trans.txt'
                                       .format(group, speaker))

            for line in open(labels_file):
                split = line.strip().split()
                file_id, word, timestamp  = split[0], split[1].lower(), split[2]
                word = word.split(',')
                timestamp = list(timestamp)
                # timestamp[0] = '-1'
                # timestamp.insert(1, ',')
                # timestamp[-1] = '-1'
                # timestamp.insert(-1, ',')
                # print(timestamp[-2])
                # print(timestamp[-3])
                # print(timestamp[-1])
                timestamp = timestamp[1:-1]
                #timestamp[-2] = ''
                timestamp = "".join(timestamp)
                timestamp = timestamp.split(',')
                timestamp = list(map(float, timestamp))

                #word = list(word)
                #word = word[1:-1]
                # timestamp[-2] = ''
                #word = "".join(word)
                #word = word.split(',')
                # word = list(map(int, word))
                words.append(word)
                split_word = [w for w in word if w in vocab]
                split_word = split_word[1:-1]
                split_word = [w for w in split_word if w != '']
                if len(split_word) == 0:
                    split_word = ['<PAD>']
                # result = bert_embedding(split_word)
                # bert_feat = np.concatenate([np.asarray(i[1]) for i in result], 0)
                # bert_feats.append(bert_feat)
                timestamps.append(timestamp)

    data_dir_360 = '/home/wangye/wangye/data/LibriSpeech/train-clean-360'
    for group in os.listdir(data_dir_360):
        speaker_path = os.path.join(data_dir_360, group)
        for speaker in os.listdir(speaker_path):
            print(speaker)
            labels_file = os.path.join(speaker_path, speaker,
                                       '{}-{}.trans.txt'
                                       .format(group, speaker))
            for line in open(labels_file):
                split = line.strip().split()
                file_id = split[0]
                label = ' '.join(split[1:]).lower()
                if file_id in unaligned_list:
                    continue
                audio_file = os.path.join(speaker_path, speaker,
                                          file_id) + '.flac'
                import librosa
                time = librosa.get_duration(filename=audio_file)
                #print(time)
                # audio = wave.open(audio_file)
                duration = float(time)
                # audio.close()
                file_ids.append(file_id)
                keys.append(audio_file)
                durations.append(duration)
                labels.append(label)

        data_dir_360_2 = '/home/wangye/wangye/data/LibriSpeech-Alignments/LibriSpeech/train-clean-360'
        speaker_path2 = os.path.join(data_dir_360_2, group)
        for speaker in os.listdir(speaker_path2):
            labels_file = os.path.join(speaker_path2, speaker,
                                       '{}-{}.alignment.txt'
                                       .format(group, speaker))

            for line in open(labels_file):
                split = line.strip().split()
                file_id, word, timestamp  = split[0], split[1].lower(), split[2]
                word = word.split(',')
                timestamp = list(timestamp)
                # timestamp[0] = '-1'
                # timestamp.insert(1, ',')
                # timestamp[-1] = '-1'
                # timestamp.insert(-1, ',')
                # print(timestamp[-2])
                # print(timestamp[-3])
                # print(timestamp[-1])
                timestamp = timestamp[1:-1]
                #timestamp[-2] = ''
                timestamp = "".join(timestamp)
                timestamp = timestamp.split(',')
                timestamp = list(map(float, timestamp))

                #word = list(word)
                #word = word[1:-1]
                # timestamp[-2] = ''
                #word = "".join(word)
                #word = word.split(',')
                # word = list(map(int, word))
                words.append(word)
                split_word = [w for w in word if w in vocab]
                split_word = split_word[1:-1]
                split_word = [w for w in split_word if w != '']
                if len(split_word) == 0:
                    split_word = ['<PAD>']
                # result = bert_embedding(split_word)
                # bert_feat = np.concatenate([np.asarray(i[1]) for i in result], 0)
                # bert_feats.append(bert_feat)
                timestamps.append(timestamp)




    lines = []
    for i in range(len(keys)):
        line = {'key': keys[i], 'duration':durations[i], 'sentence': words[i]}
        lines.append(json.dumps(line))
    #x = json.dumps(lines)
    with open(output_file, 'w') as out_file:
        out_file.write(json.dumps(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str,default='/home/wangye/wangye/data/LibriSpeech/train-clean-100',
                        help='Path to data directory')
    parser.add_argument('output_file', type=str,default='train_corpus_460.json',
                        help='Path to output file')
    args = parser.parse_args()
    main(args.data_directory, args.output_file)