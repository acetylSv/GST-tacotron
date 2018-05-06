import sys, os
import tensorflow as tf
import numpy as np
import random
from random import shuffle

from network import *
from hyperparams import Hyperparams as hp
from utils import *
#from data_load import *
from load_tfrecords import *

# Define Graph
class Graph:
    def __init__(self, mode="train"):
        # Set phase
        is_training=True if mode=="train" else False
        
        # Input, Output Placeholder
        # x: int seq. [batch_size, seq_len]
        # y: Reduced mel [batch_size, seq_len//r, n_mels*r]
        # x, z: mag [batch_size, seq_len, 1+n_fft//2]
        if mode=='train':
            #self.x, self.y, self.z, self.fnames, self.num_batch = get_batch()
            self.x, self.y, self.z, self.fnames, self.num_batch = get_batch(mode)
        elif mode=='infer':
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels*hp.r))
        else:
            self.x = tf.placeholder(tf.int32, shape=(None, None))
            self.y = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels*hp.r))
            self.z = tf.placeholder(tf.float32, shape=(None, None, 1+hp.n_fft//2))
            self.fnames = tf.placeholder(tf.string, shape=(None,))

        self.encoder_inputs = embed(self.x, len(hp.char_set), hp.embed_size)
        self.decoder_inputs = tf.concat((tf.zeros_like(self.y[:, :1, :]), self.y[:, :-1, :]), 1)
        self.decoder_inputs = self.decoder_inputs[:, :, -hp.n_mels:] # feed last frames only (N, Ty/r, n_mels)
        
        # Network
        with tf.variable_scope("ref_encoder"):
            # ref_emb = [batch, hp.ref_enc_gru_size]
            self.ref_emb = build_ref_encoder(self.y, is_training)
        
        with tf.variable_scope("STL_layer"):
            # style_emb = [batch, hp.token_emb_size]
            self.style_emb, self.GST = build_STL(self.ref_emb)

        with tf.variable_scope("encoder"):
            # memory = [batch_size, seq_len, 2*gru_size=embed_size]
            self.memory, self.encoder_final_state = build_encoder(self.encoder_inputs, is_training=is_training)
            # fusing style embedding into encoder outputs for decoder's attention
            seq_len = tf.shape(self.x)[1]
            self.memory += tf.tile(tf.expand_dims(self.style_emb, axis=1), [1, seq_len, 1])

        with tf.variable_scope("decoder1"):
            # y_hat =  [batch_size, seq_len//r, n_mels*r]
            self.y_hat, self.alignments = build_decoder1(
                    self.decoder_inputs, self.memory, self.encoder_final_state, is_training=is_training
                )

        with tf.variable_scope("decoder2"):
            # z_hat = [batch_size, seq_len, (1+n_fft//2)]
            self.z_hat = build_decoder2(self.y_hat, is_training=is_training)

        '''
        print(self.x.shape, self.y.shape, self.z.shape)
        print(self.y_hat.shape, self.z_hat.shape)
        print(self.encoder_inputs.shape, self.decoder_inputs.shape)
        exit()
        '''
        '''
        vs = [v for v in tf.trainable_variables()]
        for v in vs : print(v)
        exit()
        '''

        if mode in ("train", "eval"):
            # monitor
            self.audio_hat = tf.py_func(spectrogram2wav, [self.z_hat[0]], tf.float32)
            self.audio_gt = tf.py_func(spectrogram2wav, [self.z[0]], tf.float32)
            # Loss
            self.loss1 = tf.reduce_mean(tf.abs(self.y_hat - self.y))
            self.loss2 = tf.reduce_mean(tf.abs(self.z_hat - self.z))
            self.loss = self.loss1 + self.loss2
            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.lr = learning_rate_decay(hp.lr, global_step=self.global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            # gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_norm(grad, 5.)
                self.clipped.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(
                                self.clipped,
                                global_step=self.global_step
                            )
            # Summary
            tf.summary.scalar('{}/loss1'.format(mode), self.loss1)
            tf.summary.scalar('{}/loss2'.format(mode), self.loss2)
            tf.summary.scalar('{}/loss'.format(mode), self.loss)
            tf.summary.scalar('{}/lr'.format(mode), self.lr)
            tf.summary.image("{}/mel_gt".format(mode),
                       tf.expand_dims(self.y, -1), max_outputs=1)
            tf.summary.image("{}/mel_hat".format(mode),
                       tf.expand_dims(self.y_hat, -1), max_outputs=1)
            tf.summary.image("{}/mag_gt".format(mode),
                       tf.expand_dims(self.z, -1), max_outputs=1)
            tf.summary.image("{}/mag_hat".format(mode),
                       tf.expand_dims(self.z_hat, -1), max_outputs=1)
            tf.summary.audio("{}/sample_hat".format(mode),
                       tf.expand_dims(self.audio_hat, 0), hp.sr)
            tf.summary.audio("{}/sample_gt".format(mode),
                       tf.expand_dims(self.audio_gt, 0), hp.sr)
            self.summary_op = tf.summary.merge_all()

            # init
            self.init_op = tf.global_variables_initializer()

if __name__ == '__main__':
    g = Graph(mode='train'); print('Graph Test OK')
