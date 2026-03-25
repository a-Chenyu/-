import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 1. 准备训练数据 ----------------------
# 示例语料（可替换为真实文本，每行一个句子，分词后列表）
corpus = [
    ["apple", "banana", "fruit", "red", "sweet"],
    ["cat", "dog", "animal", "pet", "play"],
    ["python", "java", "language", "code", "program"],
    ["apple", "fruit", "tree", "leaf", "green"],
    ["cat", "meow", "animal", "soft", "cute"],
    ["python", "data", "analysis", "numpy", "pandas"],
]

# ---------------------- 2. 训练Word2Vec模型 ----------------------
# sg=1 表示使用Skip-gram算法，vector_size=5 词向量维度
model = Word2Vec(
    sentences=corpus,
    vector_size=5,
    window=2,
    min_count=1,
    sg=1,
    epochs=100
)

# 保存模型
model.save("word2vec_model.model")

# ---------------------- 3. 提取词向量并可视化 ----------------------
# 选取10个目标词
target_words = ["apple", "banana", "cat", "dog", "python", "java", "fruit", "animal", "code", "tree"]
# 提取词向量
word_vectors = np.array([model.wv[word] for word in target_words])

# 降维（5维 -> 2维，便于可视化）
tsne = TSNE(n_components=2, random_state=42, perplexity=3)
word_vectors_2d = tsne.fit_transform(word_vectors)

# 绘图
plt.figure(figsize=(8, 6))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], c="blue", edgecolors="black")

# 标注每个词
for i, word in enumerate(target_words):
    plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

plt.title("Word2Vec 词向量可视化（t-SNE降维）")
plt.savefig("word_vectors_visualization.png")
plt.show()
