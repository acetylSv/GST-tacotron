import sys, os
import tensorflow as tf
import numpy as np
from utils import *
import codecs
import re
import unicodedata

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.char_set)}
    idx2char = {idx: char for idx, char in enumerate(hp.char_set)}
    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn') # Strip accents
    text = text.lower()
    text = re.sub("[^{}]".format(hp.char_set), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode_and_write(writer, fpath, text):
    fname, mel, mag = load_spectrograms(fpath)
    example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'mel_raw': _bytes_feature(mel.tostring()),
                        'mag_raw': _bytes_feature(mag.tostring()),
                        'frame_length': _int64_feature(len(mag)),
                        'wav_filename': _bytes_feature(tf.compat.as_bytes(fname)),
                        'text_length': _int64_feature(len(text)),
                        'text': _bytes_feature(np.array(text, dtype=np.float32).tostring())
                    })
              )
    writer.write(example.SerializeToString())
    return

def make_tfrecords(mode):
    # check save dir exists
    if not os.path.exists(hp.feat_path):
        os.makedirs(hp.feat_path)
    
    output_filepath = os.path.join(hp.feat_path, mode + '.tfrecords')
    # check file
    if os.path.exists(output_filepath):
        print('=====Tfrecords %s exists, nothing new written=====' % output_filepath)
        return

    # write TFRecords Object
    writer = tf.python_io.TFRecordWriter(output_filepath)
    
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    lines = codecs.open(hp.transcript_path, 'r', 'utf-8').readlines()
    if mode=="train":
        lines = lines[32:]
    else:
        lines = lines[:31]

    for line in lines:
        if hp.EM_dataset:
            # EM dataset parsing
            fname = line.strip().split(' ')[1]
            text = ' '.join(line.strip().split(' ')[3:])
        else:
            # LJ dataset parsing
            fname, _, text = line.strip().split('|')
        fpath = os.path.join(hp.data_path, fname + ".wav")
        text = text_normalize(text) + "E"  # E: EOS
        text = [char2idx[char] for char in text]

        encode_and_write(writer, fpath, text)

    writer.close()
    return

def main():
    # extract training feat
    mode = 'train'
    make_tfrecords(mode)
    
    # extract eval feat
    mode = 'eval'
    make_tfrecords(mode)
    return

if __name__ == '__main__':
    main()
