from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import os

def read_corpus(file_path):
    """读取语料并分词"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield simple_preprocess(line.strip())

def train_word2vec(corpus_path, model_path):
    """训练Word2Vec模型"""
    corpus = list(read_corpus(corpus_path))
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,    # 词向量维度
        window=5,           # 窗口大小
        min_count=1,        # 最小词频
        workers=4,
        epochs=20
    )
    model.save(model_path)
    print(f"模型已保存至 {model_path}")
