import tensorflow as tf

from hyperparams import Hyperparams as hp

def embed(inputs, vocab_size, num_units, zero_pad=True):
    '''Embeds a given tensor.'''
    with tf.variable_scope('embedding'):
        lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, \
                shape=[vocab_size, num_units], \
                initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01) \
                )
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
    return tf.nn.embedding_lookup(lookup_table, inputs)

def prenet(inputs, is_training):
    outputs = tf.layers.dense(inputs, units=hp.prenet1_size,
                        activation=tf.nn.relu, name="dense1")
    outputs = tf.layers.dropout(outputs, rate=hp.prenet_dropout_rate,
                        training=is_training, name="dropout1")
    outputs = tf.layers.dense(outputs, units=hp.prenet2_size,
                        activation=tf.nn.relu, name="dense2")
    outputs = tf.layers.dropout(outputs, rate=hp.prenet_dropout_rate,
                        training=is_training, name="dropout2")
    return outputs

def gru(inputs, bidirection, num_units=None):
    if num_units == None:
        num_units = hp.gru_size
    cell = tf.contrib.rnn.GRUCell(num_units)
    if bidirection:
        cell_bw = tf.contrib.rnn.GRUCell(num_units)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(
                        cell, cell_bw, inputs,
                        dtype=tf.float32
                    )
        return tf.concat(outputs, 2), tf.concat(state, 1)
    else:
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        return outputs, state

def attention_decoder(inputs, memory, initial_state, num_units=None):
    if num_units == None:
        num_units = inputs.get_shape().as_list[-1]
    # inputs = [batch, seq_len, prenet2_size]
    # memory = [batch, seq_len, gru_size]
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, memory)
    decoder_cell = tf.contrib.rnn.GRUCell(num_units)
    cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(
                            decoder_cell,
                            attention_mechanism,
                            num_units,
                            alignment_history=True
                        )
    #decoder_initial_state = cell_with_attention.zero_state(dtype=tf.float32, batch_size=tf.shape(inputs)[0])
    decoder_initial_state = cell_with_attention.zero_state(dtype=tf.float32, batch_size=hp.batch_size)
    decoder_initial_state = decoder_initial_state.clone(cell_state=initial_state)
    outputs, state = tf.nn.dynamic_rnn(
                        cell_with_attention,
                        inputs,
                        #initial_state=decoder_initial_state,
                        dtype=tf.float32
                    )

    return outputs, state

def bn(inputs, is_training, activation_fn=None):
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims

    # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
    # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
    if inputs_rank in [2, 3, 4]:
        if inputs_rank == 2:
            inputs = tf.expand_dims(inputs, axis=1)
            inputs = tf.expand_dims(inputs, axis=2)
        elif inputs_rank == 3:
            inputs = tf.expand_dims(inputs, axis=1)

        outputs = tf.contrib.layers.batch_norm(
                    inputs=inputs,
                    center=True, scale=True, updates_collections=None,
                    is_training=is_training, fused=True,
                  )

        # restore original shape
        if inputs_rank == 2:
            outputs = tf.squeeze(outputs, axis=[1, 2])
        elif inputs_rank == 3:
            outputs = tf.squeeze(outputs, axis=1)

    else:  # fallback to naive batch norm
        outputs = tf.contrib.layers.batch_norm(
                    inputs=inputs,
                    center=True, scale=True, updates_collections=None,
                    is_training=is_training, fused=False)

    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return outputs

def instance_norm(inputs):
    axis = [1,2] # for format: NHWC
    epsilon = 1e-5
    mean, var = tf.nn.moments(inputs, axis, keep_dims=True)
    outputs = (inputs - mean) / tf.sqrt(var+epsilon)

    return outputs

def conv1d(inputs, filters=None, size=1, dilation=1,
           padding="SAME", use_bias=False, activation_fn=None):
    if padding.lower()=="causal":
        # pre-padding for causality
        pad_len = (size - 1) * dilation  # padding size
        inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
        padding = "valid"

    if filters is None:
        filters = inputs.get_shape().as_list[-1]

    params = {"inputs":inputs, "filters":filters, "kernel_size":size,
              "dilation_rate":dilation, "padding":padding,
              "activation":activation_fn, "use_bias":use_bias}

    outputs = tf.layers.conv1d(**params)

    return outputs

def conv1d_banks(inputs, K, is_training):
    '''Applies a series of conv1d separately.

    Args:
    inputs: A 3d tensor with shape of [N, T, C]
    K: An int. The size of conv1d banks. That is,
        The `inputs` are convolved with K filters: 1, 2, ..., K.
    is_training: A boolean. This is passed to an argument of `bn`.
    Returns:
        A 3d tensor with shape of [N, T, K*Hp.conv1d_filter_size///2]. '''

    outputs = conv1d(inputs, hp.conv1d_filter_size//2, 1) # k=1
    for k in range(2, K+1): # k = 2...K
        with tf.variable_scope("num_{}".format(k)):
            output = conv1d(inputs, hp.conv1d_filter_size//2, k)
            outputs = tf.concat((outputs, output), -1)

    outputs = bn(outputs, is_training=is_training, activation_fn=tf.nn.relu)

    return outputs # (N, T, Hp.embed_size//2*K)

def conv2d(inputs, filters=None, size=[1,1], dilation=[1,1], strides=[1,1],
           padding="SAME", use_bias=True, activation_fn=None):
    if padding.lower()=="causal":
        # pre-padding for causality
        pad_len = (size - 1) * dilation  # padding size
        inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [pad_len, 0]])
        padding = "valid"

    if filters is None:
        filters = inputs.get_shape().as_list[-1]

    params = {"inputs":inputs, "filters":filters,
              "kernel_size":size, "strides":strides,
              "dilation_rate":dilation, "padding":padding,
              "activation":None, "use_bias":use_bias}

    outputs = tf.layers.conv2d(**params)

    return outputs

def GLU(inputs):
    with tf.variable_scope('GLU'):
        c_size = inputs.get_shape()[-1]
        conv_w = conv2d(inputs[:,:,:,:c_size//2], filters=c_size//2, size=[3,3])
        conv_v = conv2d(inputs[:,:,:,c_size//2:], filters=c_size//2, size=[3,3])
        outputs = conv_w * tf.sigmoid(conv_v)

    return outputs

def highwaynet(inputs, num_units=None):
    '''Highway networks, see https://arxiv.org/abs/1505.00387
     Args:
        inputs: A 3D tensor of shape [N, T, W].
        num_units: An int or `None`. Specifies the number of units in the
        highway layer or uses the input size if `None`.
    Returns:
        A 3D tensor of shape [N, T, W].'''
    if not num_units:
        num_units = inputs.get_shape()[-1]
    H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
    T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
        bias_initializer=tf.constant_initializer(-1.0), name="dense2")
    outputs = H*T + inputs*(1.-T)

    return outputs

def multi_head_attention(query, value, num_heads=8, attention_type='mlp_attention',
                         num_units=None, normalize=True):
    ''' ref https://github.com/syang1993/gst-tacotron/blob/master/models/multihead_attention.py '''
    def _split_last_dimension(inputs):
        static_dim = inputs.get_shape().as_list()
        dynamic_dim = tf.shape(inputs)
        assert static_dim[-1] % hp.num_heads == 0
        return tf.reshape(inputs, [dynamic_dim[0], dynamic_dim[1], hp.num_heads, static_dim[-1] // hp.num_heads])
    def _split_heads(q, k, v):
        # qs = [batch_size, num_heads, 1, num_unit//num_heads]
        # ks = [batch_size, num_heads, token_num, num_unit//num_heads]
        # vs = [batch_size, num_heads, token_num, hp.token_emb_size//num_heads]
        qs = tf.transpose(_split_last_dimension(q), [0, 2, 1, 3])
        ks = tf.transpose(_split_last_dimension(k), [0, 2, 1, 3])
        vs = tf.tile(tf.expand_dims(v, axis=1), [1, hp.num_heads, 1, 1])
        return qs, ks, vs

    def _dot_product(qs, ks, vs, num_units):
        # qk = [batch_size, num_heads, 1, token_num]
        qk = tf.matmul(qs, ks, transpose_b=True)
        scale_factor = (num_units // hp.num_heads)**-0.5
        if hp.attn_normalize:
            qk *= scale_factor
        # weights = [batch_size, num_heads, 1, token_num]
        weights = tf.nn.softmax(qk, name="dot_attention_weights")
        # context = [batch_size, num_heads, 1, hp.token_emb_size//num_heads]
        context = tf.matmul(weights, vs)
        return context
    def _mlp_attention(qs, ks, vs):
        num_units = qs.get_shape()[-1].value
        v = tf.get_variable("attention_v", [num_units], dtype=qs.dtype)
        if hp.attn_normalize:
            # Scalar used in weight normalization
            g = tf.get_variable(
                    "attention_g", dtype=qs.dtype,
                    initializer=tf.sqrt((1. / num_units))
                )
            # Bias added prior to the nonlinearity
            b = tf.get_variable(
                    "attention_b", [num_units], dtype=qs.dtype,
                    initializer=tf.zeros_initializer()
                )
            # normed_v = g * v / ||v||
            normed_v = g * v * tf.rsqrt(tf.reduce_sum(tf.square(v)))
            add = tf.reduce_sum(normed_v * tf.tanh(ks + qs + b), [-1], keep_dims=True)
        else:
            add = tf.reduce_sum(v * tf.tanh(ks + qs), [-1], keep_dims=True)
        
        # weights = [batch_size, num_heads, 1, token_num]
        weights = tf.nn.softmax(tf.transpose(add, [0, 1, 3, 2]), name="mlp_attention_weights")
        # context = [batch_size, num_heads, 1, hp.token_emb_size//num_heads]
        context = tf.matmul(weights, vs)
        
        return context

    if num_units is None:
        num_units = query.get_shape().as_list()[-1]
    if num_units % hp.num_heads != 0:
        raise ValueError("Multi head attention requires that num_units is a multiple of {}".format(num_heads))

    q = tf.layers.conv1d(query, num_units, 1)
    k = tf.layers.conv1d(value, num_units, 1)
    v = value
    qs, ks, vs = _split_heads(q, k, v)
    if attention_type == 'mlp_attention':
        style_emb = _mlp_attention(qs, ks, vs)
    elif attention_type == 'dot_attention':
        style_emb = _dot_product(qs, ks, vs, num_units)
    else:
        raise ValueError('Only mlp_attention and dot_attention are supported')

    # combine each head to one
    style_emb = tf.reshape(style_emb, [hp.batch_size, hp.token_emb_size])

    return style_emb
