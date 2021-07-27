import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr
import re

# if sys.version_info[0] > 2:
#     is_py3 = True
# else:
#     reload(sys)
#     sys.setdefaultencoding("utf-8")
#     is_py3 = False
#
#
# def native_word(word, encoding='utf-8'):
#     """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
#     if not is_py3:
#         return word.encode(encoding)
#     else:
#         return word

#
# def native_content(content):
#     if not is_py3:
#         return content.decode('utf-8')
#     else:
#         return content

#
# def open_file(filename, mode='r'):
#     """
#     常用文件操作，可在python2和python3间切换.
#     mode: 'r' or 'w' for read or write
#     """
#     if is_py3:
#         return open(filename, mode, encoding='utf-8', errors='ignore')
#     else:
#         return open(filename, mode)

def remove_1a(content):
    # 去除标点字母数字
    chinese = '[\u4e00-\u9fa5a-zA-Z0-9]+'
    str1 = re.findall(chinese, content)
    return ''.join(str1)

def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    content=remove_1a(content)
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word=dict(zip(range(len(words)),words))
    return words, word_to_id,id_to_word


def read_category():
    """读取分类目录，固定"""
    # categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    # categories = [native_content(x) for x in categories]
    # cat_to_id = dict(zip(categories, range(len(categories))))
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id={'体育':0, '财经':1, '房产':2, '家居':3, '教育':4, '科技':5, '时尚':6, '时政':7, '游戏':8, '娱乐':9}
    id_to_cat={0:'体育',1:'财经', 2:'房产', 3:'家居', 4:'教育', 5:'科技', 6:'时尚', 7:'时政', 8:'游戏', 9:'娱乐'}

    return categories, cat_to_id, id_to_cat


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
