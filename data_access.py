import json
from hyperparams import Hyperparams as hp
import tensorflow as tf
from keras.preprocessing import sequence


# 1 读取字典
def get_X_Y_dictional():
    with open('./Data_1_Pinyin2Sample_chinese/Map_Pinyin_and_Samplechinese.json', 'r', encoding='utf-8')as f:
        r_d = json.load(f)
    return r_d['W2I_X'], r_d['I2W_X'], r_d['W2I_Y'], r_d['I2W_Y']


# 2 读取数据集'trian,eval

def split_train_eval():
    with open('./Data_1_Pinyin2Sample_chinese/Pinyin_and_Samplechinese_original.tsv', 'r', encoding='utf-8') as f1:
        all_line = f1.readlines()
    f_train = []
    f_eval = []

    for i, line in enumerate(all_line):
        if i % 50 == 1:
            f_eval.append(line)
        else:
            f_train.append(line)
    return f_train, f_eval


# 3 生成训练数据
def get_batch_data():
    # 读入数据
    all_line_train, _ = split_train_eval()
    XX_W2I, XX_I2W, YY_W2I, YY_I2W = get_X_Y_dictional()

    X_line = [line.replace('\n', '').split('\t')[1] for line in all_line_train]
    Y_line = [line.replace('\n', '').split('\t')[2] for line in all_line_train]

    x_list = [[XX_W2I.get(word, 1) for word in (source_sent + u" </S>").split()] for source_sent in X_line]
    y_list = [[YY_W2I.get(word, 1) for word in (target_sent + u" </S>").split()] for target_sent in Y_line]

    # padding
    X = tf.convert_to_tensor(sequence.pad_sequences(x_list, maxlen=hp.maxlen, padding='post'), tf.int32)
    Y = tf.convert_to_tensor(sequence.pad_sequences(y_list, maxlen=hp.maxlen, padding='post'), tf.int32)

    # 随机打乱，创建新的批次
    input_queues = tf.train.slice_input_producer([X, Y])
    x, y = tf.train.shuffle_batch(input_queues,
                                  num_threads=8,
                                  batch_size=hp.batch_size,
                                  capacity=hp.batch_size * 64,
                                  min_after_dequeue=hp.batch_size * 32,
                                  allow_smaller_final_batch=False)
    num_batch = len(x_list) // hp.batch_size
    return x, y, num_batch


# 4 生成评价数据
def get_eval_data():
    _____, all_line_eval = split_train_eval()
    XX = []
    YY = []

    for line in all_line_eval:
        line_temp = line.replace('\n', '').split('\t')
        XX.append(line_temp[1])
        YY.append(line_temp[2])

    XX_W2I, _, __, ___ = get_X_Y_dictional()

    x_list = []
    for source_sent in XX:
        x = [XX_W2I.get(word, 1) for word in (source_sent + u" </S>").split()]
        x_list.append(x)

    X = sequence.pad_sequences(x_list, maxlen=hp.maxlen, padding='post')
    return X, XX, YY
