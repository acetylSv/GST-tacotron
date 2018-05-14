# GST-tacotron
Reproducing:
Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis
(https://arxiv.org/pdf/1803.09017.pdf)

## Python and Toolkit Version
    Python:      '3.5.2'
    numpy:       '1.13.1'
    tensorflow:  '1.4'
## Steps and Usages
1. Data Preprocess:
    - Prepare wavs and transcription
    - Set path informations in hyperparams.py
2. Make TFrecords for faster data loading:
    - Check path, partition informations in hyperparams.py
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
    - Run:
<pre><code>python3 infer.py [ref_audio_path]</code></pre>

## Notes
At experiments 6-1 and 6-2, paper points out that one could SELECT some tokens, scale it and then feed this style embedding into text encoder. But at section 3.2.2, multi-head attention is used and each token is set to be 256/h dim. 
If so, at inference time, a selected token should have 256/h dim, but the text encoder should be fused with a 256 dim vector. And also, if one choose to use multi-head attention, then the style embedding will become some vectors'(which is the attention result of each head) concatenation passing through a linear network rather than GST's weighed sum. I do not really understand that if this is the case, can one simply SELECT some GST's combination to represent style embedding or not.

## TODO
- [x] Find an adequate dataset (Blizzard 2013)
- [ ] Implement feed_previous function in decoder RNN
- [ ] Input phone seqs instead of character seqs
- [ ] WaveNet vocoder