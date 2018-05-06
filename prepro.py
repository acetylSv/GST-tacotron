''' ref: https://www.github.com/kyubyong/tacotron '''
from utils import load_spectrograms
from hyperparams import Hyperparams as hp
import os
from data_load import load_data
import numpy as np
import tqdm

# Load data
fpaths, _, _ = load_data() # list

if not os.path.exists(hp.feat_path): os.mkdir(hp.feat_path)
if not os.path.exists("{}/mels".format(hp.feat_path)): os.mkdir("{}/mels".format(hp.feat_path))
if not os.path.exists("{}/mags".format(hp.feat_path)): os.mkdir("{}/mags".format(hp.feat_path))

for fpath in fpaths:
    res = fpath.split('/')[-1].replace('wav', 'npy')
    if not os.path.exists("{}/mels/{}".format(hp.feat_path, res)) or not \
           os.path.exists("{}/mags/{}".format(hp.feat_path, res)):
        print(fpath)
        fname, mel, mag = load_spectrograms(fpath)
        np.save("{}/mels/{}".format(hp.feat_path, fname.replace("wav", "npy")), mel)
        np.save("{}/mags/{}".format(hp.feat_path, fname.replace("wav", "npy")), mag)
