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
    total_len = len(df)
    chunk_size = total_len // n_chunks + (total_len % n_chunks > 0)  # 计算每个块的大小
    
    for i in range(n_chunks):
        chunk_start = i * chunk_size
        # 确保最后一个块能包含所有剩余的行
        chunk_end = min((i + 1) * chunk_size, total_len)
        chunk_df = df.iloc[chunk_start:chunk_end]
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

# 定义一个函数来加载和合并指定的chunks
def load_and_concat_chunks(chunk_paths1, chunk_paths2):
    combined_chunks = []
    for path1, path2 in zip(chunk_paths1, chunk_paths2):
        df1 = pd.read_pickle(path1)
        df2 = pd.read_pickle(path2)
        combined_chunk = pd.concat([df1, df2], ignore_index=True)
        combined_chunks.append(combined_chunk)
    return combined_chunks

# 定义保存chunks的函数
def save_chunks(chunks, output_dir, label):
    for i, chunk in enumerate(chunks):
        chunk.to_pickle(os.path.join(output_dir, f"{label}_chunk{i+1}.pkl"))

# 加载并合并real data和non_dy的对应chunks
data_chunks = load_and_concat_chunks(real_data_paths, none_dy_paths)

# 保存合并后的数据chunks
save_chunks(data_chunks, output_dir_base, 'data')


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

data_chunk_paths = ['/vols/cms/yl13923/masterproject/data_chunk1.pkl',
                     '/vols/cms/yl13923/masterproject/data_chunk2.pkl', 
                     '/vols/cms/yl13923/masterproject/data_chunk3.pkl', 
                     '/vols/cms/yl13923/masterproject/data_chunk4.pkl', 
                     '/vols/cms/yl13923/masterproject/data_chunk5.pkl']
mc_chunk_paths = ['/vols/cms/yl13923/masterproject/DY_chunk1.pkl', 
                  '/vols/cms/yl13923/masterproject/DY_chunk2.pkl', 
                  '/vols/cms/yl13923/masterproject/DY_chunk3.pkl', 
                  '/vols/cms/yl13923/masterproject/DY_chunk4.pkl', 
                  '/vols/cms/yl13923/masterproject/DY_chunk5.pkl']

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