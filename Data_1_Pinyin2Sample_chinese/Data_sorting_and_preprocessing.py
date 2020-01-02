import json


# 0 sorting环节
def data_sorting():
    # 主要生成标准格式，[X1,X2,X3,X4,X5,X6,X7,...    Y1,Y2,Y3,Y4,Y5,Y6,Y7,...]
    return True


# 1 生成字典
def generate_speech_dict():
    # 读入数据
    with open('./Pinyin_and_Samplechinese_original.tsv', 'r', encoding='utf-8') as f1:
        all_line = f1.readlines()

    X_line = []
    Y_line = []
    for line in all_line:
        line_temp = line.replace('\n', '').split('\t')
        X_line.append(line_temp[1])
        Y_line.append(line_temp[2])
    # x 字统计
    word_statistics_X = {}
    for txt in X_line:
        for speech in txt.split(' '):
            if len(speech) > 1:
                word_statistics_X[speech] = '1'
    word_counts_X = list(word_statistics_X.keys())
    W2I_X = {'<PAD>': 0, '<UNK>': 1, '<S>': 2, '</S>': 3}
    I2W_X = {0: "<PAD>", 1: "<UNK>", 2: "<S>", 3: "</S>"}
    for idx, word in enumerate(word_counts_X):
        W2I_X[word] = idx + 4
        I2W_X[idx + 4] = word

    # y 字统计
    word_statistics_Y = {}
    for txt in Y_line:
        for speech in txt.split(' '):
            word_statistics_Y[speech] = '2'
    word_counts_Y = list(word_statistics_Y.keys())

    W2I_Y = {'<PAD>': 0, '<UNK>': 1, '<S>': 2, '</S>': 3}
    I2W_Y = {0: '<PAD>', 1: '<UNK>', 2: '<S>', 3: '</S>'}
    for idx, word in enumerate(word_counts_Y):
        W2I_Y[word] = idx + 4
        I2W_Y[idx + 4] = word

    output_data = {}
    output_data['W2I_X'] = W2I_X
    output_data['I2W_X'] = I2W_X
    output_data['W2I_Y'] = W2I_Y
    output_data['I2W_Y'] = I2W_Y
    with open('./Map_Pinyin_and_Samplechinese.json', 'w', encoding='utf-8')as ff:
        json.dump(output_data, ff, ensure_ascii=False)

