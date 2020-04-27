import re

import jieba
from gensim.models.fasttext import FastText
from jieba import cut
from tqdm import tqdm


class Segmentation:

    def __init__(self):
        """
        loading dict and stopwords
        """
        jieba.load_userdict('./data/dict.txt')
        with open('./data/stopwords.txt', 'r') as f:
            self.stopwords = [re.sub(r'\n', '', word)
                              for word in f.readlines()]

    def execute(self, text):
        # remain only Chinese
        text = re.sub(r'\W|[0-9a-zA-Z]', '', text)
        words = []
        for word in cut(text):
            if word not in self.stopwords:
                words.append(word)
        return list(filter(None, words))


class Embedding:
    def __init__(self, path):
        self.path = path

    def train_vec(self, dataset):
        """
        training word embedding

        Args:
            dataset (2D array): 2D tokens
        """
        self.model = FastText(dataset, min_count=1, workers=8)
        self.model.save(self.path)

    def infer_vec(self, dataset):
        """
        get vector

        Args:
            dataset (2D array): 2D tokens

        Returns:
            output (2D array): [[[vec, vec...], label], [[vec, vec...], label]...]
        """
        self.model = FastText.load(self.path)

        output = []
        for tokens, label in tqdm(dataset):
            w2v = []
            for token in tokens:
                try:
                    w2v.append(self.model.wv[token])
                except:
                    continue
            if not w2v:
                continue
            output.append([w2v, label])
        return output
