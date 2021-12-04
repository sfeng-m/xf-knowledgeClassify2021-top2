
import random
import pandas as pd
import jieba
import pickle
import os
from copy import copy
import translators as ts
from gensim.models.word2vec import Word2Vec

from dataProcess.train_w2v import w2v_main, load_stopwords

import io, sys
#改变标准输出的默认编码
# sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

def translator_constructor(api):
    if api == 'google':
        return ts.google
    elif api == 'bing':
        return ts.bing
    elif api == 'baidu':
        return ts.baidu
    elif api == 'sogou':
        return ts.sogou
    elif api == 'youdao':
        return ts.youdao
    elif api == 'tencent':
        return ts.tencent
    elif api == 'alibaba':
        return ts.alibaba
    else:
        raise NotImplementedError(f'{api} translator is not realised!')


class SampleExpand():
    def __init__(self) -> None:
        super(SampleExpand, self).__init__()
        self.w2v_model_path = './code/dataProcess/save_w2v_model/w2v_model.md'
        self.train_data = pd.read_csv("./datasets/train_data.csv")
        self.train_data['Analysis'] = self.train_data['Analysis'].fillna('')
        self.senlen_threshold = 16
        self.sample_maxlen = 512
        self.random_name = ['random_add', 'random_del', 'random_replace', 'random_swap', 'random_puncuation']
        self.change_ratio = 0.15
        self.stopwords = load_stopwords()
        if not os.path.exists(self.w2v_model_path):
            w2v_main(self.train_data)
        self.model = Word2Vec.load(self.w2v_model_path)
        self.API = 'google'
        self.PUNCTUATIONS = ['，','。','？','！','：','；']
        self.PUNC_RATIO = 0.3

    def expand_sample(self, sentence):
        """扩增样本主函数，包括随机增删改及同义词替换等方法"""
        if self.is_senlen_enough(sentence):
            rand_name = random.choice(self.random_name)
            if rand_name == 'random_add':
                return self.random_add(sentence)
            elif rand_name == 'random_del':
                return self.random_del(sentence)
            elif rand_name == 'random_replace':
                return self.random_replace(sentence)
            elif rand_name == 'random_swap':
                return self.random_swap(sentence)
            elif rand_name == 'translate':
                return self.translate(sentence)
            elif rand_name == 'random_puncuation':
                return self.insert_punctuation_marks(sentence)
        else:
            return sentence

    def truncate_sample(self, sample):
        """对样本进行截断处理"""

    def is_senlen_enough(self, sentence):
        """句子长度是否满足要求（太短的句子不进行增删改）
        """
        return True if len(sentence) > self.senlen_threshold else False

    def random_add(self, sentence):
        """对一个句子进行随机添加一个词，添加的词来自于词表
        """
        words = jieba.lcut(sentence)
        new_words = words.copy()
        add_num = int(len(words)*self.change_ratio)
        for _ in range(add_num):
            counter = 0
            while True:
                random_word = new_words[random.randint(0, len(new_words)-1)]
                if random_word in self.model.wv.vocab:
                    random_synonym = self.model.similar_by_word(random_word)[0][0]
                    random_idx = random.randint(0, len(new_words)-1)
                    new_words.insert(random_idx, random_synonym)
                    break
                counter += 1
                if counter >= 10:
                    break
        return ''.join(new_words)

    def random_del(self, sentence):
        """对一个句子进行随机删除一个词(当句子中某个词出现多次时，一起删除)
        """
        words = jieba.lcut(sentence)
        #obviously, if there's only one word, don't delete it
        if len(words) == 1:
            return sentence
        #randomly delete words with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > self.change_ratio:
                new_words.append(word)
        #if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return words[rand_int]
        return ''.join(new_words)

    def random_replace(self, sentence):
        """对一个句子随机选择一个词进行替换，替换的词通过词向量相似度计算获取(当句子中某个词出现多次时，一起替换)
        """
        words = jieba.lcut(sentence)
        new_words = words.copy()
        random_word_list = [word for word in words if word not in self.stopwords and word in self.model.wv.vocab]
        random.shuffle(random_word_list)
        replace_num = int(len(random_word_list)*self.change_ratio)
        num_replaced = 0
        for random_word in random_word_list:
            similar_words = self.model.similar_by_word(random_word)[:2]
            # similar_words = [w[0] for w in similar_words if w[1] >= 0.9]
            if len(similar_words) >= 1:
                sim_word = random.choice(similar_words)[0]
                new_words = [sim_word if word == random_word else word for word in new_words]
                #print("replaced", random_word, "with", synonym)
                num_replaced += 1
            if num_replaced >= replace_num: #only replace up to n words
                break
        return ''.join(new_words)

    def random_swap(self, sentence):
        words = jieba.lcut(sentence)
        new_words = words.copy()
        swap_num = int(len(words)*self.change_ratio)
        for _ in range(swap_num):
            random_idx_1 = random.randint(0, len(new_words)-1)
            random_idx_2 = random_idx_1
            counter = 0
            while random_idx_2 == random_idx_1:
                random_idx_2 = random.randint(0, len(new_words)-1)
                counter += 1
                if counter > 3:
                    break
            new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
        return ''.join(new_words)

    def translate(self, text):
        r = random.uniform(0, 1)
        if r > 0.5:
            trans_lang = 'en'
        else:
            trans_lang = random.choice(['fr', 'de', 'ja', 'es'])
        try:
            trans_text = translator_constructor(self.API)(text, 'zh', trans_lang)
            return translator_constructor(self.API)(trans_text, trans_lang, 'zh')
        except:
            return text

    # Insert punction words into a given sentence with the given ratio "punc_ratio"
    def insert_punctuation_marks(self, sentence):
        words = jieba.lcut(sentence)
        new_line = []
        q = random.randint(1, int(self.PUNC_RATIO * len(words) + 1))
        qs = random.sample(range(0, len(words)), q)

        for j, word in enumerate(words):
            if j in qs:
                new_line.append(self.PUNCTUATIONS[random.randint(0, len(self.PUNCTUATIONS)-1)])
                new_line.append(word)
            else:
                new_line.append(word)
        return ''.join(new_line)


def eda_data(data, repeat_n=5):
    """使用增删换等进行数据增强
    """
    EDA = SampleExpand()
    result = data.copy(deep=True)
    for n in range(repeat_n):
        new_data = data.copy(deep=True)
        new_data['Content'] = new_data['Content'].apply(lambda x: EDA.expand_sample(x))
        new_data['Analysis'] = new_data['Analysis'].apply(lambda x: EDA.expand_sample(x))
        new_data['options'] = new_data['options'].apply(lambda x: EDA.expand_sample(x))
        result = pd.concat([result, new_data])
    return result


def translate(text):
    API = 'google'
    trans_text = translator_constructor(API)(text, 'zh', 'es')
    return translator_constructor(API)(trans_text, 'es', 'zh')


if __name__ == '__main__':
    text = "本题考核电子商务法的基本制度。电子商务法的基本制度包括电子商务合同法律制度、电子签名和电子认证法律制度、电子支付法律制度。"
    print(translate(text))