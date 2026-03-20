from gensim.models import Word2Vec

# 读取数据
def load_data(path):
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sentences.append(line.strip().split())
    return sentences

def train():
    sentences = load_data("data/sample.txt")

    # 训练word2vec模型
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )

    # 保存模型
    model.save("model/word2vec.model")
    print("模型训练完成并保存！")

if __name__ == "__main__":
    train()
