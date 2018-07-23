# GST-tacotron
Reproducing:
Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis
(https://arxiv.org/pdf/1803.09017.pdf)

## Python and Toolkit Version
    Python:      '3.5.2'
    numpy:       '1.13.1'
    tensorflow:  '1.4'
    
## Samples and Pretrained Models
Samples could be found [here](./samples), where two kind of experiments were conducted:
1. Conditioning on reference audio:
    * BZ_440K.wav is an inference result from model trained on Blizzard2013 for 440K steps (batch_size=16), the conditioned referecne audio is picked from its testing set.
    * LJ_448K.wav is another inference result from model trained on LJ_Speech for 448K steps (batch_size=16), the conditioned referecne audio is also picked from its testing set.
2. Combinations of GSTs:
    * normal.wav and slow.wav are two inference results from model trained on LJ_Speech, the difference between the two is by picking difference style tokens for style embedding.
    * high.wav and low.wav is another pair of example.
Pretrained models on both datasets could be downloaded [here](http://speech.ee.ntu.edu.tw/~acetylsv/pretrained_model.zip).
Download pretrained models and set path in hyperparams.py, then you can use infer.py to generate speech.
<strong>Note that the detailed settings are different and listed below:</strong>
1. pretrained_model_BZ:
	* n_fft:1024
	* sample_rate: 16000
2. pretrained_model_LJ:
	* n_fft:2048
	* sample_rate: 22050

## Steps and Usages
1. Data Preprocess:
    - Prepare wavs and transcription
    - Example format:
<pre><code>Blizzard_2013|CA-MP3-17-138.wav|End of Mansfield Park by Jane Austin.
Blizzard_2013|CA-MP3-17-139.wav|Performed by Catherine Byers.
...</code></pre>
2. Make TFrecords for faster data loading:
    - Check parameters in hyperparams.py
        - path informations
        - TFrecords partition number
        - sample_rate
        - fft points, hop length, window length
        - mel-filter banks number
    - Run:
<pre><code>python3 make_tfrecords.py</code></pre>
3. Train the whole network:
    - Check log directory and model, summary settings in hyperparams.py
    - Run:
<pre><code>python3 train.py</code></pre>
4. Evaluation while training:
    - (Currently only do evaluation on the first batch_size data)
    - (Decoder RNN are manually "FEED_PREVIOUS" now)
    - Run:
<pre><code>python3 eval.py</code></pre>
5. Inference:
    - Check Inference input text in hyperparams.py
    - Pass reference audio path as argument
    - Reference audio: an arbitary .wav file
    - Directly condition on combination of GSTs is now undergoing, set below flag in infer.py <code>condition_on_audio = False</code> and set the combination weight you like
    - Run:
<pre><code>python3 infer.py [ref_audio_path]</code></pre>
6. Inference input text example format:
<pre><code>0. Welcome to N. T. U. speech lab
1. Recognize speech
2. Wreck a nice beach
...</code></pre>

## Notes
1. At experiments 6-1 and 6-2, paper points out that one could SELECT some tokens, scale it and then feed this style embedding into text encoder. But at section 3.2.2, multi-head attention is used and each token is set to be 256/h dim. 
If so, at inference time, a selected token should have 256/h dim, but the text encoder should be fused with a 256 dim vector. And also, if one choose to use multi-head attention, then the style embedding will become some vectors'(which is the attention result of each head) concatenation passing through a linear network rather than GST's weighed sum. I do not really understand that if this is the case, can one simply SELECT some GST's combination to represent style embedding or not.
2. Input phone seqs did not give better results or faster convergence speed.
3. Dynamic bucketing with feed_previous decoder RNN seems not possible?
   (tf.split, tf.unstack, tf.shape, get_shape().as_list(), slicing... not seems
   to work)
## TODO
- [x] Find an adequate dataset (Blizzard 2013)
- [x] (Failed) Implement feed_previous function in decoder RNN
- [x] Input phone seqs instead of character seqs
- [ ] WaveNet vocoder
