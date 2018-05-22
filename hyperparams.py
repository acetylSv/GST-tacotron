class Hyperparams:
    #### Path Info ####
    #log_dir = './tacotron_log_LJ_5_16_480K_converge'
    log_dir = './tacotron_log_BZ_max300_5_16'
    data_path = './data'
    transcript_path = './data/transcription_blizzard2013_max300.txt'
    feat_path = './feat_BZ_max300'
    sample_dir = './sample_dir'
    infer_data_path = './harvard_sentences.txt'

    #### Char-set ####
    char_set = "PE abcdefghijklmnopqrstuvwxyz'.?" # P for padding, E for eos

    #### Data Loading ####
    tfrecords_partition = 100

    #### Modules ####
    ## tacotron-1
    embed_size = 256
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4

    prenet1_size = 256
    prenet2_size = 128 # prenet2_size should be hp.embed_size//2 for Residual Connections
    prenet_dropout_rate = 0.5

    conv1d_filter_size = 256
    gru_size = 256
    
    ## reference encoder
    ref_enc_filters = [32, 32, 64, 64, 128, 128]
    ref_enc_size = [3,3]
    ref_enc_strides = [2,2]
    ref_enc_gru_size = 128

    ## style token layer
    token_num = 10
    token_emb_size = 256
    num_heads = 8
    multihead_attn_num_unit = 128
    style_att_type = 'mlp_attention'
    attn_normalize = True

    #### Networks ####
    lr = 0.001
    batch_size = 16
    summary_period = 300
    save_period = 1000
    r = 5 # Reduction factor. Paper => 2, 3, 5

    #### Signal Processing ####
    is_trimming = True
    sr = 16000 # Sample rate.
    n_fft = 1024 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples.
    win_length = int(sr*frame_length) # samples.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 200 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20
