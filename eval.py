import sys, os
import tensorflow as tf
import numpy as np

from network import *
from hyperparams import Hyperparams as hp
from utils import *
from graph import Graph

def eval():
    # Build graph
    g = Graph(mode='eval'); print("Eval Graph loaded")
    # Load data
    fpaths, text_lengths, texts = load_data(mode="eval")
    # Parse
    text = np.fromstring(texts[0], np.int32)
    fname, mel, mag = load_spectrograms(fpaths[0])
    
    x = np.expand_dims(text, 0) # (1, None)
    y = np.expand_dims(mel, 0) # (1, None, n_mels*r)
    z = np.expand_dims(mag, 0) # (1, None, 1+n_fft//2)

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
    # Summary
    summary_writer = tf.summary.FileWriter(hp.log_dir, sess.graph)
    
    # Feed Forward
    ## mel
    infer_y_hat = np.zeros((1, y.shape[1], y.shape[2]), np.float32)  # hp.n_mels*hp.r
    for j in range(y.shape[1]):
        a, temp_y_hat = sess.run([g.decoder_inputs, g.y_hat], {g.x: x, g.y: infer_y_hat})
        infer_y_hat[:, j, :] = temp_y_hat[:, j, :]
    ## mag
    summary_str, gs, al = sess.run(
                            [g.summary_op, g.global_step, g.alignments],
                            {g.x:x, g.y:y, g.y_hat: infer_y_hat, g.z: z}
                        )
    summary_writer.add_summary(summary_str, gs)
    plot_alignment(al[0], gs, mode='eval')

    # exit
    summary_writer.close()
    sess.close()

if __name__ == '__main__':
    eval()
    print('Eval Done')
