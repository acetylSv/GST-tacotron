import sys, os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from network import *
from hyperparams import Hyperparams as hp
from utils import *
from graph import Graph

# init random_seed
#tf.set_random_seed(2401)
#np.random.seed(2401)
#random.seed(2401)

def train():
    # Build graph
    g = Graph(mode='train'); print("Training Graph loaded")
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
        print("=====Init a new model=====")
        sess.run([g.init_op])
        gs = 0
    # Summary
    summary_writer = tf.summary.FileWriter(hp.log_dir, sess.graph)

    try:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #import time
        #tS = time.time()
        while True:
            for _ in range(g.num_batch):
                if coord.should_stop():
                    break
                _, loss, gs = sess.run([g.train_op, g.loss, g.global_step])
                print('===GS:  %s, loss:  %lf===' % (str(gs), loss))
                #if gs == 80:
                #    tE = time.time()
                #    print("It cost %f sec" % (tE - tS))
                #    exit()
                if(gs % hp.summary_period == 0):
                    summary_str, al = sess.run([g.summary_op, g.alignments])
                    # add summ
                    summary_writer.add_summary(summary_str, gs)
                    # plot alignment
                    plot_alignment(al[0], gs, mode='train')

                if(gs % hp.save_period == 0):
                    saver.save(sess, os.path.join(hp.log_dir, 'model.ckpt'), global_step=gs)
                    print('Save model to %s-%d' % (os.path.join(hp.log_dir, 'model.ckpt'), gs))

    except Exception as e:
        coord.request_stop(e)
    finally :
        coord.request_stop()
        coord.join(threads)

    # exit
    summary_writer.close()
    sess.close()

if __name__ == '__main__':
    train()
    print('Training Done')
