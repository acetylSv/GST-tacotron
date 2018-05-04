import sys, os
import tensorflow as tf
import numpy as np
from scipy.io.wavfile import write

from network import *
from hyperparams import Hyperparams as hp
from utils import *
from data_load import *
from graph import Graph

def infer():
    # Build graph
    g = Graph(mode='infer'); print("Infer Graph loaded")
    # Load data
    texts = load_data(mode="infer")

    # Saver
    saver = tf.train.Saver(max_to_keep = 5)
    # Session
    sess = tf.Session()
    # If model exist, restore, else init a new one
    ckpt = tf.train.get_checkpoint_state(hp.log_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("=====Reading model parameters from %s=====" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        print("=====Reading model error=====")
        exit()

    # Feed Forward
    ## mel
    y_hat = np.zeros((texts.shape[0], 200, hp.n_mels*hp.r), np.float32)  # hp.n_mels*hp.r
    for j in range(200):
        _y_hat = sess.run(g.y_hat, {g.x: texts, g.y: y_hat})
        y_hat[:, j, :] = _y_hat[:, j, :]
    ## mag
    mags, al = sess.run([g.z_hat, g.alignments], {g.x: texts, g.y:y_hat, g.y_hat:y_hat})

    for i, mag in enumerate(mags):
        print("File {}.wav is being generated ...".format(i+1))
        audio = spectrogram2wav(mag)
        write(os.path.join(hp.sample_dir, '{}.wav'.format(i+1)), hp.sr, audio)
        plot_alignment(al[i], gs, mode='infer')

    # exit
    sess.close()

if __name__ == '__main__':
    if not os.path.exists(hp.sample_dir): os.mkdir(hp.sample_dir)
    infer()
    print('Inference Done')
