import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

# 加载模型
model = Word2Vec.load("model/word2vec.model")

# 取前10个词
words = list(model.wv.index_to_key)[:10]
vectors = [model.wv[word] for word in words]

# PCA降维
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

# 画图
plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    x, y = result[i]
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, word)

plt.title("Word2Vec词向量可视化")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
