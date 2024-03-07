import pandas as pd
import matplotlib.pyplot as plt
import os

# 加载数据块的函数
def load_data_chunks(directory, prefix=""):
    all_data = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(".pkl"):
            file_path = os.path.join(directory, filename)
            chunk_data = pd.read_pickle(file_path)
            all_data = pd.concat([all_data, chunk_data], ignore_index=True)
    return all_data

# 加载数据
combined_data = load_data_chunks("combined_chunks", "combined_chunk")
MC_data = load_data_chunks("MC_chunks", "MC_chunk")


# 绘制Z_mass的分布图，加入权重
plt.figure(figsize=(10, 6))
plt.hist(combined_data['Z_mass'], bins=50, range=[50,200], weights=combined_data['wt'], alpha=0.5, label='Combined', histtype='step')
plt.hist(MC_data['Z_mass'], bins=50, range=[50,200], weights=MC_data['wt'], alpha=0.5, label='MC', histtype='step')
plt.xlabel('Z Mass (GeV/c^2)')
plt.ylabel('Events')
plt.title('Z Mass Distribution')
plt.legend()
plt.savefig("Z Mass Distribution.png")

# 绘制Z_pt的分布图，加入权重
plt.figure(figsize=(10, 6))
plt.hist(combined_data['Z_pt'], bins=50, range=[50,200], weights=combined_data['wt'], alpha=0.5, label='Combined', histtype='step')  # 添加权重
plt.hist(MC_data['Z_pt'], bins=50, range=[50,200], weights=MC_data['wt'], alpha=0.5, label='MC', histtype='step')  # 添加权重
plt.xlabel('Z $p_T$ (GeV/c)')
plt.ylabel('Events')
plt.title('Z $p_T$ Distribution')
plt.legend()
plt.savefig("Z_pt Distribution.png")
