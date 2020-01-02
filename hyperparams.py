# 超参数
class Hyperparams:
    batch_size = 32  # alias = N
    lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'log'  # log directory

    # model
    maxlen = 100  # Maximum number of words in a sentence. alias = T.
    num_blocks = 6  # number of encoder/decoder blocks
    num_heads = 8
    hidden_units = 512  # alias = C
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    num_epochs = 50
    dropout_rate = 0.1
