from hyperparams import Hyperparams as hp
from utils import *

import numpy as np
import tensorflow as tf
import codecs

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
                    serialized_example,
                    features={
                        'mel_raw': tf.FixedLenFeature([], tf.string),
                        'mag_raw': tf.FixedLenFeature([], tf.string),
                        'frame_length': tf.FixedLenFeature([], tf.int64),
                        'wav_filename': tf.FixedLenFeature([], tf.string),
                        'text': tf.FixedLenFeature([], tf.string),
                        'text_length': tf.FixedLenFeature([], tf.int64),
                    }
                )
    get_mel = tf.decode_raw(features['mel_raw'], tf.float32)
    get_mag = tf.decode_raw(features['mag_raw'], tf.float32)
    get_text = tf.decode_raw(features['text'], tf.float32)
    get_frame_length = tf.cast(features['frame_length'], tf.int32)
    get_wav_filename = features['wav_filename']
    get_text_length = tf.cast(features['text_length'], tf.int32)

    get_mel = tf.reshape(get_mel, [get_frame_length//hp.r, hp.n_mels*hp.r])
    get_mag = tf.reshape(get_mag, [get_frame_length, 1+hp.n_fft//2])
    get_text = tf.cast(tf.reshape(get_text, [get_text_length]), tf.int32)
    return get_mel, get_mag, get_wav_filename, get_text_length, get_text

def get_batch(mode):
    def _get_max_min_len():
        lines = codecs.open(hp.transcript_path, 'r', 'utf-8').readlines()
        if hp.EM_dataset:
            # EM dataset parsing
            text_lengths = [len(' '.join(line.strip().split(' ')[3:])) for line in lines]
        else:
            # LJ dataset parsing
            text_lengths = [len(line.strip().split('|')[-1]) for line in lines]
        return max(text_lengths), min(text_lengths), len(text_lengths) // hp.batch_size

    # create queue
    filename_queue = tf.train.string_input_producer([os.path.join(hp.feat_path, mode+'.tfrecords')])
    get_mel, get_mag, get_wav_filename, get_text_length, get_text = read_and_decode(filename_queue)
    maxlen, minlen, num_batch = _get_max_min_len()
    
    # Batching
    _, (texts, mels, mags, wav_filenames) = \
            tf.contrib.training.bucket_by_sequence_length(
                input_length=get_text_length,
                tensors=[get_text, get_mel, get_mag, get_wav_filename],
                batch_size=hp.batch_size,
                bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                num_threads=16,
                capacity=hp.batch_size * 4,
                dynamic_pad=True
             )
            #16, 4 = 142secs 32, 8 = 138secs, 16, 1 = 135secs
    return texts, mels, mags, wav_filenames, num_batch
