import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
def visualize_words(model_path, words):    
"""可视化指定词的词向量分布"""
    model = Word2Vec.load(model_path)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    valid_words = [word for word in words if word in model.wv]
    # PCA降维到2D
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(word_vectors)
    # 绘图
    plt.figure(figsize=(10, 8))    
for i, word in enumerate(valid_words):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
        plt.text(vectors_2d[i, 0]+0.01, vectors_2d[i, 1]+0.01, word, fontsize=12)
    plt.title("Word2Vec词向量分布可视化")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()
