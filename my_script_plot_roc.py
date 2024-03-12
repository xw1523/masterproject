import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_loss_curve(history_path, save_path):
    with open(history_path, 'r') as file:
        history = json.load(file)
    
    train_loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, 'b-o', label='Training loss')
    plt.plot(epochs, val_loss, 'g--o', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)  # 保存图表
    plt.close()  # 关闭当前绘图，以便开始下一个

# 指定一个历史数据文件路径和对应的保存路径
history_paths = [
    '/vols/cms/yl13923/masterproject/shuffled_chunk1_v2_history.json',
    '/vols/cms/yl13923/masterproject/shuffled_chunk2_v2_history.json',
    '/vols/cms/yl13923/masterproject/shuffled_chunk3_v2_history.json',
    '/vols/cms/yl13923/masterproject/shuffled_chunk4_v2_history.json',
    '/vols/cms/yl13923/masterproject/shuffled_chunk5_v2_history.json',
]

# 这里指定每个图表的保存路径
save_loss_paths = [
    'chunk1_loss_curve_v2.png',
    'chunk2_loss_curve_v2.png',
    'chunk3_loss_curve_v2.png',
    'chunk4_loss_curve_v2.png',
    'chunk5_loss_curve_v2.png',
]

for history_path, save_path in zip(history_paths, save_loss_paths):
    plot_loss_curve(history_path, save_path)

def plot_combined_roc_curves(roc_data_paths, save_path):
    plt.figure(figsize=(8, 6))
    auc_scores = []  # 存储每个数据块的AUC分数
    
    for path in roc_data_paths:
        with open(path, 'r') as file:
            roc_data = json.load(file)
        fpr, tpr, auc_score = roc_data['fpr'], roc_data['tpr'], roc_data['auc_score']
        plt.plot(fpr, tpr, label=f'Chunk {roc_data_paths.index(path) + 1} (AUC = {auc_score:.3f})')
        auc_scores.append(auc_score)

    plt.plot([0, 1], [0, 1], 'k--', label='Chance level')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves')
    plt.legend(loc='best')
    plt.savefig(save_path)  # 保存图表
    plt.close()  # 关闭当前绘图

# 定义包含所有ROC数据文件路径的列表
roc_data_paths = [
    '/vols/cms/yl13923/masterproject/shuffled_chunk1_v2_roc_data.json',
    '/vols/cms/yl13923/masterproject/shuffled_chunk2_v2_roc_data.json',
    '/vols/cms/yl13923/masterproject/shuffled_chunk3_v2_roc_data.json',
    '/vols/cms/yl13923/masterproject/shuffled_chunk4_v2_roc_data.json',
    '/vols/cms/yl13923/masterproject/shuffled_chunk5_v2_roc_data.json',
]

# 调用函数，保存聚合ROC曲线图到指定路径
plot_combined_roc_curves(roc_data_paths, 'combined_roc_curves_v2.png')
