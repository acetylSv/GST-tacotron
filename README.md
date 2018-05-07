# GST-tacotron
Reproducing Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis (https://arxiv.org/pdf/1803.09017.pdf)

## Python and Toolkit Version
	Python:      '3.5.2'
	numpy:       '1.13.1'
	tensorflow:  '1.4'

## Usage
	make_tfrecords.py: python3 make_tfrecords.py
		Extracting features and transorm to TFRecords for training
		(Path and dataset informations are in hyperparams.py)
	train.py: python3 train.py
		Training stage
	eval.py: python3 eval.py
		Evaluating stage
		(Currently only eval on the first batch_size)
	infer.py: python3 infer.py <ref_audio_path>
		Inferring stage
		(Text inputs are in hyperparams.py now)


## TODO
- [ ] Find an adequate dataset (Blizzard 2013)
- [ ] Input phone seqs instead of character seqs
- [ ] WaveNet vocoder
- [ ] Evaluation data loading
