<<<<<<< HEAD
import os
import pandas as pd
import uproot
import numpy as np
import json

# 从文件名提取数据集名称的函数
def dataset_name_func(file_path):
    return file_path.split('/')[-1].split('_')[0]

# 获取缩放因子的函数
def get_scale_factor(params, dataset, lumi=59830):
    data = params.get(dataset, None)
    if data:
        xs = data.get('xs', 1)
        evt = data.get('evt', 1)
        return xs * lumi / evt
    return 1

# 处理文件并分块保存为.pkl文件的函数
def process_and_save_combined_chunks(file_paths, dataset_name_func, output_dir, params, n_chunks=5, chunk_prefix="", is_background=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    combined_data = pd.DataFrame()

    for file_path in file_paths:
        dataset_name = dataset_name_func(file_path)
        scale_factor = get_scale_factor(params, dataset_name)
        
        with uproot.open(file_path) as file:
            tree = file["ntuple"]
            data = tree.arrays(['Z_mass', 'Z_pt', 'wt'], library="pd")
            data['wt'] *= scale_factor

            # 如果是背景样本，乘以-1
            if chunk_prefix == "background":    
                data['wt'] *= -1
        
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    # 随机打乱所有合并后的数据
    combined_data = combined_data.sample(frac=1).reset_index(drop=True)

    # 分块处理并保存
    total_len = len(combined_data)
    chunk_size = total_len // n_chunks + (total_len % n_chunks > 0)

    for i in range(n_chunks):
        chunk_start = i * chunk_size
        chunk_end = min((i + 1) * chunk_size, total_len)
        chunk = combined_data.iloc[chunk_start:chunk_end]
        
        chunk_filename = os.path.join(output_dir, f"{chunk_prefix}_chunk_{i}.pkl")
        chunk.to_pickle(chunk_filename)


# 加载 JSON 文件
with open('/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/params_UL_2018.json') as f:
    params = json.load(f)



# 示例：处理背景样本
background_samples = [
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/Tbar-tW_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo1L3Nu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/T-tW_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W4JetsToLNu-LO_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W1JetsToLNu-LO_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/EWKZ2Jets_ZToLL_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/EWKWMinus2Jets_WToLNu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WGToLNuG_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/EWKWPlus2Jets_WToLNu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WWTo2L2Nu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo3LNu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo1L1Nu2Q_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WJetsToLNu-LO_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/Tbar-t_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WWTo1L1Nu2Q_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W3JetsToLNu-LO_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo2Q2L_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W2JetsToLNu-LO_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/ZZTo4L_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/T-t_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/ZZTo2L2Nu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/TTTo2L2Nu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/TTToHadronic_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/TTToSemiLeptonic_zmm_2018.root"
]

MC_samples=[
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY2JetsToLL-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DYJetsToLL_M-10to50-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY3JetsToLL-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY4JetsToLL-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DYJetsToLL-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DYJetsToLL-LO-ext1_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY1JetsToLL-LO_zmm_2018.root"]

data_samples=["/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonC_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonA_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonB_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonD_zmm_2018.root"]

output_dir_bg = "background_chunks"
output_dir_MC = "MC_chunks"
output_dir_data = "data_chunks"


process_and_save_combined_chunks(background_samples, dataset_name_func, output_dir_bg, params, n_chunks=5, chunk_prefix="background")
process_and_save_combined_chunks(MC_samples, dataset_name_func, output_dir_MC, params, n_chunks=5, chunk_prefix="MC")
process_and_save_combined_chunks(data_samples, dataset_name_func, output_dir_data, params, n_chunks=5, chunk_prefix="data")


####combine background and data
import pandas as pd
import os

def combine_chunks(background_dir, data_dir, output_dir, n_chunks):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(n_chunks):
        background_chunk_path = os.path.join(background_dir, f"background_chunk_{i}.pkl")
        data_chunk_path = os.path.join(data_dir, f"data_chunk_{i}.pkl")
        
        # 读取背景和数据文件
        background_chunk = pd.read_pickle(background_chunk_path)
        data_chunk = pd.read_pickle(data_chunk_path)
        
        # 合并背景和数据文件
        combined_chunk = pd.concat([background_chunk, data_chunk], ignore_index=True)
        
        # 随机打乱合并后的数据
        combined_chunk = combined_chunk.sample(frac=1).reset_index(drop=True)
        
        # 保存合并后的数据块
        combined_chunk_path = os.path.join(output_dir, f"combined_chunk_{i}.pkl")
        combined_chunk.to_pickle(combined_chunk_path)

# 调用函数
background_dir = "background_chunks"
data_dir = "data_chunks"
output_dir = "combined_chunks"
n_chunks = 5  # 假设你有5个背景和数据文件

combine_chunks(background_dir, data_dir, output_dir, n_chunks)

combine_chunks=['/vols/cms/xw1523/masterproject/combined_chunks/combined_chunk_0.pkl',
'/vols/cms/xw1523/masterproject/combined_chunks/combined_chunk_1.pkl',
'/vols/cms/xw1523/masterproject/combined_chunks/combined_chunk_2.pkl',
'/vols/cms/xw1523/masterproject/combined_chunks/combined_chunk_3.pkl',
'/vols/cms/xw1523/masterproject/combined_chunks/combined_chunk_4.pkl']

mc_chunk=['/vols/cms/xw1523/masterproject/MC_chunks/MC_chunk_0.pkl','/vols/cms/xw1523/masterproject/MC_chunks/MC_chunk_1.pkl','/vols/cms/xw1523/masterproject/MC_chunks/MC_chunk_2.pkl','/vols/cms/xw1523/masterproject/MC_chunks/MC_chunk_3.pkl','/vols/cms/xw1523/masterproject/MC_chunks/MC_chunk_4.pkl']
=======
import uproot3
import pandas as pd
import numpy as np
import json
import os

def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def process_files_to_dataframe(file_paths, json_data, lum=59830, is_dy=False, adjust_weight=True):
    branches = ['Z_mass', 'Z_pt', 'wt']
    combined_df = pd.DataFrame()

    for file_path in file_paths:
        json_file_name = file_path.split('/')[-1].split('_zmm_2018')[0]
        xs = evt = 1  # Default values for real data
        if adjust_weight and json_file_name in json_data:  # Adjust weight only for MC data
            xs = json_data[json_file_name]['xs']
            evt = json_data[json_file_name]['evt']

        with uproot3.open(file_path) as file:
            df = file['ntuple'].pandas.df(branches)
            if adjust_weight:  # Apply weight adjustment only if needed
                df['wt'] *= xs * lum / evt
                if not is_dy:
                    df['wt'] = -df['wt']
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df

def split_and_save_dataframe(df, output_dir, label, n_chunks=5):
    chunk_size = len(df) // n_chunks
    for i in range(n_chunks):
        chunk_df = df[i*chunk_size : (i+1)*chunk_size if i < n_chunks - 1 else None]
        chunk_df.to_pickle(os.path.join(output_dir, f"{label}_chunk{i+1}.pkl"))



json_data = read_json('/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/params_UL_2018.json')

None_DY_list = [
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/Tbar-tW_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo1L3Nu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/T-tW_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W4JetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W1JetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/EWKZ2Jets_ZToLL_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/EWKWMinus2Jets_WToLNu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WGToLNuG_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/EWKWPlus2Jets_WToLNu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WWTo2L2Nu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo3LNu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo1L1Nu2Q_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WJetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/Tbar-t_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WWTo1L1Nu2Q_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W3JetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo2Q2L_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W2JetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/ZZTo4L_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/T-t_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/ZZTo2L2Nu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/TTTo2L2Nu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/TTToHadronic_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/TTToSemiLeptonic_zmm_2018.root"]


DY_list = [
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY2JetsToLL-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DYJetsToLL_M-10to50-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY3JetsToLL-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY4JetsToLL-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DYJetsToLL-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DYJetsToLL-LO-ext1_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY1JetsToLL-LO_zmm_2018.root'
]


Real_Data_list = [
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonC_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonA_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonB_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonD_zmm_2018.root',
]



output_dir_base = '/vols/cms/yl13923/masterproject'

# Process and save each category
for category, file_paths in [('None_DY', None_DY_list), ('DY', DY_list), ('Real_Data', Real_Data_list)]:
    # 对于实际数据(Real_data)，设置 adjust_weight 为 False，否则为 True
    adjust_weight = category != 'Real_Data'
    df = process_files_to_dataframe(file_paths, json_data, is_dy=(category == 'DY'), adjust_weight=adjust_weight)
    split_and_save_dataframe(df, output_dir_base, category)

# 合并混合样本并分块保存

# 加载处理后的数据块
none_dy_paths = [os.path.join(output_dir_base, f'None_DY_chunk{i+1}.pkl') for i in range(5)]
dy_paths = [os.path.join(output_dir_base, f'DY_chunk{i+1}.pkl') for i in range(5)]
real_data_paths = [os.path.join(output_dir_base, f'Real_Data_chunk{i+1}.pkl') for i in range(5)]

# print("none_dy_paths:", none_dy_paths)
# print("dy_paths:", dy_paths)
# print("real_data_paths:", real_data_paths)

def load_and_combine_pkl(file_paths):
    """加载多个Pickle文件并将它们合并为一个DataFrame"""
    df_list = [pd.read_pickle(path) for path in file_paths]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

none_dy_df = load_and_combine_pkl(none_dy_paths)
dy_df = load_and_combine_pkl(dy_paths)
real_data_df = load_and_combine_pkl(real_data_paths)

# 合并实际数据和非DY模拟数据为“data”，DY保留为“MC”
data_df = pd.concat([real_data_df, none_dy_df], ignore_index=True)
mc_df = dy_df

# 分块并保存混合样本
split_and_save_dataframe(data_df, output_dir_base, 'data', n_chunks=5)
split_and_save_dataframe(mc_df, output_dir_base, 'MC', n_chunks=5)


def label_and_combine_chunks(data_chunk_path, mc_chunk_path, label_data=1, label_mc=0):
    # 加载数据块，为其添加标签
    data_chunk = pd.read_pickle(data_chunk_path)
    data_chunk['label'] = label_data
    
    # 加载MC块，为其添加标签
    mc_chunk = pd.read_pickle(mc_chunk_path)
    mc_chunk['label'] = label_mc
    
    # 合并数据和MC块
    combined_chunk = pd.concat([data_chunk, mc_chunk], ignore_index=True)
    
    # 随机打乱合并后的数据
    shuffled_chunk = combined_chunk.sample(frac=1).reset_index(drop=True)
    
    return shuffled_chunk

data_chunk_paths = ['/vols/cms/yl13923/masterproject/data_chunk1.pkl', '/vols/cms/yl13923/masterproject/data_chunk2.pkl', '/vols/cms/yl13923/masterproject/data_chunk3.pkl', '/vols/cms/yl13923/masterproject/data_chunk4.pkl', '/vols/cms/yl13923/masterproject/data_chunk5.pkl']
mc_chunk_paths = ['/vols/cms/yl13923/masterproject/MC_chunk1.pkl', '/vols/cms/yl13923/masterproject/MC_chunk2.pkl', '/vols/cms/yl13923/masterproject/MC_chunk3.pkl', '/vols/cms/yl13923/masterproject/MC_chunk4.pkl', '/vols/cms/yl13923/masterproject/MC_chunk5.pkl']

for i, (data_path, mc_path) in enumerate(zip(data_chunk_paths, mc_chunk_paths), start=1):
    # 处理每对数据和MC块，添加标签，合并并随机打乱
    shuffled_chunk = label_and_combine_chunks(data_path, mc_path)
    
    # 保存处理后的数据块
    shuffled_chunk.to_pickle(f'/vols/cms/yl13923/masterproject/shuffled_chunk{i}.pkl')

shuffled_chunk_paths = [
    '/vols/cms/yl13923/masterproject/shuffled_chunk1.pkl',
    '/vols/cms/yl13923/masterproject/shuffled_chunk2.pkl',
    '/vols/cms/yl13923/masterproject/shuffled_chunk3.pkl',
    '/vols/cms/yl13923/masterproject/shuffled_chunk4.pkl',
    '/vols/cms/yl13923/masterproject/shuffled_chunk5.pkl'
]
>>>>>>> refs/remotes/origin/master
