
    
    print(f"展示的词：{words}")
    
    # 获取向量
    vectors = np.array([model.wv[word] for word in words])
    
    # PCA降维
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # 绘图
    plt.figure(figsize=(12, 10))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, c='lightblue', edgecolors='black', alpha=0.7)
    
    for i, word in enumerate(words):
        plt.annotate(word, 
                    xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    
    plt.title('Word2Vec词向量分布', fontsize=16, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"可视化图片已保存：{save_path}")


if __name__ == "__main__":
    visualize_model()
