from translate_chinese_english.fan_2_jian.langconv import *
from translate_chinese_english.hyperparams import Hyperparams as hp
import copy
import jieba
import json
import tensorflow as tf
from keras.preprocessing import sequence


# 0 杂乱数据生成句子对
def Data_pre_arrangement():
    """
    原始数据-->（切分好）句子对

    """

    def Traditional2Simplified(sentence):
        sentence = Converter('zh-hans').convert(sentence)
        return sentence

    path_data = r'./0_cmn_all.txt'
    with open(path_data, 'r', encoding='utf-8')as f:
        txt_list = f.readlines()

    symbol_list = [',', ':', '.', '?', '!', ')', '(', '-', '“', '”', '’', '‘',
                   '/', '"', '\'', '\\', '–', ';', '[', ']', '—', '…', '@', '#',
                   '$', '&', '*', '_', '=']
    num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '%']

    all_line = []
    all_line_sequnce = []
    for i, txt in enumerate(txt_list):
        txt_temp = txt.split('\t')

        # 英语处理
        english_line = copy.copy(txt_temp[0])
        for c in ['\n', '\t', '\xa0', '&nbsp', '\xad', '�', '\u200b', '\u3000', '\x9d']:
            english_line = english_line.replace(c, '')
        english_line = english_line.lower()
        for c in symbol_list + num_list:
            english_line = english_line.replace(c, ' ' + c + ' ')

        # 中文处理
        chinese_line = Traditional2Simplified(txt_temp[1])
        simplified_sentence = chinese_line.replace('\n', '').replace(' ', '')
        simplified_sentence_jie_cut = ' '.join(jieba.cut(simplified_sentence))

        t_temp = english_line + '\t' + simplified_sentence_jie_cut + '\n'
        all_line.append(t_temp)
        all_line_sequnce.append([english_line, simplified_sentence_jie_cut])

    with open('./A_data/english_data.tsv', 'w', encoding='utf-8') as f1:
        for line in all_line:
            f1.write(line)


# 0_1 分成训练和评价集
def split_train_eval():
    with open('./A_data/english_chinese_data.tsv', 'r', encoding='utf-8') as f1:
        all_line = f1.readlines()
    f_train = []
    f_eval = []
    for i, line in enumerate(all_line):
        if i % 20 == 1:
            f_train.append(line)
        else:
            f_eval.append(line)

    with open('./A_data/english_chinese_train.tsv', 'w', encoding='utf-8') as f11:
        for line in f_train:
            f11.write(line)

    with open('./A_data/english_chinese_test.tsv', 'w', encoding='utf-8') as f12:
        for line in f_eval:
            f12.write(line)


# 1 生成字典
def generate_speech_dict():
    # 读入数据
    with open('./A_data/english_chinese_data.tsv', 'r', encoding='utf-8') as f1:
        all_line = f1.readlines()
    english_line = []
    simplified_sentence_jie_cut = []
    for line in all_line:
        line_temp = line.replace('\n', '').split('\t')
        english_line.append(line_temp[0])
        simplified_sentence_jie_cut.append(line_temp[1])

    word_counts_chinese_kv = {}
    for txt in simplified_sentence_jie_cut:
        for speech in txt.split(' '):
            word_counts_chinese_kv[speech] = 'xx'
    word_counts_chinese = list(word_counts_chinese_kv.keys())

    word2idx_chinese = {'<PAD>': 0, '<UNK>': 1, '<S>': 2, '</S>': 3}
    idx2word_chinese = {0: "<PAD>", 1: "<UNK>", 2: "<S>", 3: "</S>"}
    for idx, word in enumerate(word_counts_chinese):
        word2idx_chinese[word] = idx + 4
        idx2word_chinese[idx + 4] = word

    word_counts_eng_kv = {}
    for txt in english_line:
        for speech in txt.split(' '):
            if len(speech) > 1:
                word_counts_eng_kv[speech] = '1'
    word_counts_eng = list(word_counts_eng_kv.keys())

    word2idx_eng = {'<PAD>': 0, '<UNK>': 1, '<S>': 2, '</S>': 3}
    idx2word_eng = {0: "<PAD>", 1: "<UNK>", 2: "<S>", 3: "</S>"}
    for idx, word in enumerate(word_counts_eng):
        word2idx_eng[word] = idx + 4
        idx2word_eng[idx + 4] = word
    return_data = {}
    return_data['word2idx_chinese'] = word2idx_chinese
    return_data['idx2word_chinese'] = idx2word_chinese
    return_data['word2idx_eng'] = word2idx_eng
    return_data['idx2word_eng'] = idx2word_eng
    with open('./A_data/english_chinese_word_dict.json', 'w', encoding='utf-8')as ff:
        json.dump(return_data, ff, ensure_ascii=False)


# 2 读取字典
def read_speech_dict():
    with open('./A_data/english_chinese_word_dict.json', 'r', encoding='utf-8')as ff:
        r_d = json.load(ff)
    return r_d['word2idx_chinese'], r_d['idx2word_chinese'], r_d['word2idx_eng'], r_d['idx2word_eng']


# 3 生成训练数据
def get_batch_data():
    # 读入数据
    train_path = r'./A_data/english_chinese_train.tsv'
    with open(train_path, 'r', encoding='utf-8') as f1:
        all_line = f1.readlines()
    english_line = []
    simplified_sentence_jie_cut = []
    for line in all_line:
        line_temp = line.replace('\n', '').split('\t')
        english_line.append(line_temp[0])
        simplified_sentence_jie_cut.append(line_temp[1])

    Y_W2I_zhong, Y_I2W_zhong, X_W2I_english, X_I2W_english = read_speech_dict()

    x_list = []
    y_list = []
    for source_sent, target_sent in zip(english_line, simplified_sentence_jie_cut):
        x = [X_W2I_english.get(word, 1) for word in (source_sent + u" </S>").split()]
        y = [Y_W2I_zhong.get(word, 1) for word in (target_sent + u" </S>").split()]
        x_list.append(x)
        y_list.append(y)
    X = sequence.pad_sequences(x_list, maxlen=hp.maxlen, padding='post')
    Y = sequence.pad_sequences(y_list, maxlen=hp.maxlen, padding='post')


    num_batch = len(X) // hp.batch_size
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    input_queues = tf.train.slice_input_producer([X, Y])
    x, y = tf.train.shuffle_batch(input_queues,
                                  num_threads=8,
                                  batch_size=hp.batch_size,
                                  capacity=hp.batch_size * 64,
                                  min_after_dequeue=hp.batch_size * 32,
                                  allow_smaller_final_batch=False)

    return x, y, num_batch


# 4 生成评价数据
def get_eval_data():
    # 读入数据
    eval_path = r'./A_data/english_chinese_eval.tsv'
    with open(eval_path, 'r', encoding='utf-8') as f1:
        all_line = f1.readlines()
    english_line = []
    simplified_sentence_jie_cut = []
    for line in all_line:
        line_temp = line.replace('\n', '').split('\t')
        english_line.append(line_temp[0])
        simplified_sentence_jie_cut.append(line_temp[1])

    Y_W2I_zhong, Y_I2W_zhong, X_W2I_english, X_I2W_english = read_speech_dict()

    x_list = []
    # y_list = []
    for source_sent, target_sent in zip(english_line, simplified_sentence_jie_cut):
        x = [X_W2I_english.get(word, 1) for word in (source_sent + u" </S>").split()]
        # y = [Y_W2I.get(word, 1) for word in (target_sent + u" </S>").split()]
        x_list.append(x)
        # y_list.append(y)
    X = sequence.pad_sequences(x_list, maxlen=hp.maxlen, padding='post')
    # Y = sequence.pad_sequences(y_list, maxlen=hp.maxlen, padding='post')
    return X, english_line, simplified_sentence_jie_cut
