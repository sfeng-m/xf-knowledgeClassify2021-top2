# -*- encoding: UTF-8 -*-

import codecs
import sys
import os
import re
import jieba.posseg as pseg
# import jieba
from gensim.models.word2vec import Word2Vec
sys.path.append('..')

# sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')


def load_stopwords():
    stopwords = codecs.open('./code/dataProcess/stopword','r',encoding='utf8').readlines()  #对问题进行分词
    stopwords = [w.strip() for w in stopwords]
    stopwords = {sw: 1 for sw in stopwords}
    return stopwords


def tokenization(text, stopwords):
    """
    分词
    :param text:
    :return:
    """
    # jieba.load_userdict('../file/user_dict')  #添加用户词典
    result = []
    words = pseg.cut(text)
    # words = jieba.lcut(text)
    for word, flag in words:
        if word not in stopwords and word != ' ':  
            result.append(word)
    return result


def word2vec_train(corpus):
    """
    word2vec模型训练
    :param corpus:
    :return:
    """
    model = Word2Vec(min_count=1)
    model.build_vocab(corpus)
    model.train(corpus, total_examples = model.corpus_count, epochs = 100)
    if not os.path.exists('./dataProcess/save_w2v_model'):
        os.makedirs('./dataProcess/save_w2v_model')
    model.save('./dataProcess/save_w2v_model/w2v_model.md')
    

def w2v_main(data):
    w2v_train_data = data['Content'].values.tolist() + data['Analysis'].values.tolist() + data['options'].values.tolist()
    w2v_train_data = list(set(w2v_train_data))
    w2v_train_data = [re.split('[。？！]', train) for train in w2v_train_data]
    w2v_train_data = [sentence for sample in w2v_train_data for sentence in sample]
    w2v_train_data = [sample.replace('\r','').replace('\n','').replace('\u3000','') for sample in w2v_train_data if sample != '']
    stopwords = load_stopwords()
    train_corpus = [tokenization(x, stopwords) for x in w2v_train_data]
    word2vec_train(train_corpus)
    
