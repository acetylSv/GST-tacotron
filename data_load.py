''' ref: https://www.github.com/kyubyong/tacotron '''
from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
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

def load_data(mode="train"):
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    if mode in ("train", "eval"):
        # Parse
        fpaths, text_lengths, texts = [], [], []
        lines = codecs.open(hp.transcript_path, 'r', 'utf-8').readlines()
        total_hours = 0
        if mode=="train":
            lines = lines[1:]
        else: # We attack only one sample!
            lines = lines[:1]
        
        for line in lines:
            if hp.EM_dataset:
                # EM dataset parsing
                fname = line.strip().split(' ')[1]
                text = ' '.join(line.strip().split(' ')[3:])
            else:
                # LJ dataset parsing
                fname, _, text = line.strip().split('|')

            fpath = os.path.join(hp.data_path, fname + ".wav")
            fpaths.append(fpath)

            text = text_normalize(text) + "E"  # E: EOS
            text = [char2idx[char] for char in text]
            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32).tostring())
        return fpaths, text_lengths, texts
    else:
        # Parse
        lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:]
        sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines] # text normalization, E: EOS
        lengths = [len(sent) for sent in sents]
        maxlen = sorted(lengths, reverse=True)[0]
        texts = np.zeros((len(sents), maxlen), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data() # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // hp.batch_size

        fpaths = tf.convert_to_tensor(fpaths, dtype=tf.string)
        text_lengths = tf.convert_to_tensor(text_lengths)
        texts = tf.convert_to_tensor(texts)

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer(
                                [fpaths, text_lengths, texts], shuffle=True
                            )
        
        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        if hp.prepro:
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath)
                mel = "{}/mels/{}".format(hp.feat_path, fname.decode('utf-8').replace("wav", "npy"))
                mag = "{}/mags/{}".format(hp.feat_path, fname.decode('utf-8').replace("wav", "npy"))
                return fname, np.load(mel), np.load(mag)
            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:
            fname, mel, mag = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])  # (None, n_mels)

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, hp.n_mels*hp.r))
        mag.set_shape((None, hp.n_fft//2+1))

        # Batching
        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=hp.batch_size,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 10)],
                                            #bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=32,
                                            capacity=hp.batch_size * 1,
                                            #capacity=hp.batch_size * 4,
                                            dynamic_pad=True)

    return texts, mels, mags, fnames, num_batch

