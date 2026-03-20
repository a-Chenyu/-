from src.train import train_word2vec
from src.visualize import visualize_words
import os
if __name__ == "__main__":    
# 确保data和models目录存在
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)        
# 训练模型
    train_word2vec(
        corpus_path="data/corpus.txt",
        model_path="models/word2vec.model"    
)
    # 选择10个词进行可视化
    target_words = ["China", "Beijing", "Shanghai", "GUangzhou", "Shenzhen", "Changsha", "Wuhan", "Yunan", "Haikou", "Tianjing"]
    visualize_words(
        model_path="models/word2vec.model",
        words=target_words    
)
