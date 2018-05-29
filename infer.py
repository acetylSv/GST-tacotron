import sys, os
import tensorflow as tf
import numpy as np
from scipy.io.wavfile import write

from network import *
from hyperparams import Hyperparams as hp
from utils import *
from graph import Graph
import make_tfrecords

def get_mel_and_mag(sess, texts, style_emb):
    # get mel
    y_hat = np.zeros((texts.shape[0], 200, hp.n_mels*hp.r), np.float32)
    for j in range(200):
        _y_hat = sess.run(g.y_hat, {g.x: texts, g.y: y_hat, g.style_emb:style_emb})
        y_hat[:, j, :] = _y_hat[:, j, :]
    # get mag
    mags, al = sess.run([g.z_hat, g.alignments], {g.x: texts, g.y:y_hat, g.y_hat:y_hat})
    return y_hat, mags, al

def infer():
    # Build graph
    g = Graph(mode='infer'); print("Infer Graph loaded")
    # Load text data
    texts = make_tfrecords.eval_infer_load_data(mode="infer")
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

    ## get style emb or use GST
    condition_on_audio = True
    if condition_on_audio:
        # Load ref audio data
        _, mel, _ = load_spectrograms(sys.argv[1])
        # ref_mel = [texts.shape[0], seq_len//hp.r, hp.n_mels]
        ref_mel = np.tile(np.expand_dims(mel, 0), (texts.shape[0], 1, 1))
        style_emb = sess.run(g.style_emb, {g.y:ref_mel})
        _, mags, al = get_mel_and_mag(sess, texts, style_emb)
        for i, mag in enumerate(mags):
            print("File {}_{}.wav is being generated ...".format(sys.argv[1].replace('.wav', ''), i+1))
            audio = spectrogram2wav(mag)
            write(os.path.join(hp.sample_dir, '{}_{}.wav'.format(sys.argv[1].replace('.wav', ''), i+1)), hp.sr, audio)
            plot_alignment(al[i], gs, i, mode='infer')
    else:
        GST = sess.run(g.GST)
        # how to pass to multi-head attention?
        GST = np.tile(GST, (1,8))
        for idx in range(10):
            scale = np.zeros(hp.token_emb_size)
            scale[:] = 0.3
            style_emb = GST[idx] * scale
            style_emb = np.tile(style_emb, (texts.shape[0], 1))
            _, mags, al = get_mel_and_mag(sess, texts, style_emb)
            for i, mag in enumerate(mags):
                print("File {}_{}.wav is being generated ...".format(idx, i+1))
                audio = spectrogram2wav(mag)
                write(os.path.join(hp.sample_dir, '{}_{}.wav'.format(idx, i+1)), hp.sr, audio)
                plot_alignment(al[i], gs, i, mode='infer')
    
    # exit
    sess.close()

if __name__ == '__main__':
    if not os.path.exists(hp.sample_dir): os.mkdir(hp.sample_dir)
    infer()
    print('Inference Done')
