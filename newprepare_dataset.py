import os
import pandas as pd
import uproot
import numpy as np
import json

def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def process_dataframe(file_paths, json_data, lum=59830, is_dy=False, adjust_weight=True):
    tree = ['Z_mass', 'Z_pt', 'wt']
    combined_df = pd.DataFrame()

    for file_path in file_paths:
        json_file_name = file_path.split('/')[-1].split('_zmm_2018')[0]
        xs = evt = 1  # Default values for real data
        if adjust_weight and json_file_name in json_data:  # Adjust weight only for MC data
            xs = json_data[json_file_name]['xs']
            evt = json_data[json_file_name]['evt']

        with uproot.open(file_path) as file:
            df = file["ntuple"].arrays(tree, library="pd") 
            if adjust_weight:  
                df['wt'] *= xs * lum / evt
                if not is_dy:
                    df['wt'] = -df['wt']
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    return combined_df


def split_dataframe(df, output_dir, label, n_chunks=5):
    total_len = len(df)
    chunk_size = total_len // n_chunks + (total_len % n_chunks > 0)  # 计算每个块的大小
    
    for i in range(n_chunks):
        chunk_start = i * chunk_size
        # 确保最后一个块能包含所有剩余的行
        chunk_end = min((i + 1) * chunk_size, total_len)
        chunk_df = df.iloc[chunk_start:chunk_end]
        chunk_df.to_pickle(os.path.join(output_dir, f"{label}_chunk{i+1}.pkl"))
        
json_data = read_json('/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/params_UL_2018.json')

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

output_dir_base = '/vols/cms/xw1523/masterproject'



# Process and save each category
for category, file_paths in [('background', background_samples), ('MC', MC_samples), ('Real_Data', data_samples)]:
    # 对于实际数据(Real_data)，设置 adjust_weight 为 False，否则为 True
    adjust_weight = category != 'Real_Data'
    df = process_dataframe(file_paths, json_data, is_dy=(category == 'MC'), adjust_weight=adjust_weight)
    split_dataframe(df, output_dir_base, category)


# 加载处理后的数据块
background_paths = [os.path.join(output_dir_base, f'background_chunk{i+1}.pkl') for i in range(5)]
MC_paths = [os.path.join(output_dir_base, f'MC_chunk{i+1}.pkl') for i in range(5)]
real_data_paths = [os.path.join(output_dir_base, f'Real_Data_chunk{i+1}.pkl') for i in range(5)]


#合并指定的chunks
def concat_chunks(chunk1, chunk2):
    combined_chunks = []
    for path1, path2 in zip(chunk1, chunk2):
        df1 = pd.read_pickle(path1)
        df2 = pd.read_pickle(path2)
        combined_chunk = pd.concat([df1, df2], ignore_index=True)
        # 随机打乱合并后的DataFrame
        shuffled_chunk = combined_chunk.sample(frac=1).reset_index(drop=True)
        # 将打乱后的数据块添加到列表中
        combined_chunks.append(shuffled_chunk)
    return combined_chunks

# 定义保存chunks的函数
def save_chunks(chunks, output_dir, label):
    for i, chunk in enumerate(chunks):
        chunk.to_pickle(os.path.join(output_dir, f"{label}_chunk{i+1}.pkl"))

# 加载并合并real data和non_dy的对应chunks
data_chunks = concat_chunks(real_data_paths, background_paths)

# 保存合并后的数据chunks
save_chunks(data_chunks, output_dir_base, 'data')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping


def process_and_save_chunk(data_chunk_path, mc_chunk_path, output_dir, chunk_index):
    # 加载数据块，为其添加标签
    data_chunk = pd.read_pickle(data_chunk_path)
    data_chunk['label'] = 1
    
    # 加载MC块，为其添加标签
    mc_chunk = pd.read_pickle(mc_chunk_path)
    mc_chunk['label'] = 0
    
    # 合并数据和MC块
    combined_chunk = pd.concat([data_chunk, mc_chunk], ignore_index=True)
    X = combined_chunk[['Z_mass', 'Z_pt']]
    y = combined_chunk['label']
    weights = combined_chunk['wt']
    
    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=0.95, random_state=42, stratify=y.values
    )
    
    # 应用MinMaxScaler进行缩放
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 将训练集和测试集保存为Pandas DataFrame
    train_df = pd.DataFrame(X_train_scaled, columns=['Z_mass', 'Z_pt'])
    train_df['label'] = y_train.reset_index(drop=True)
    train_df['wt'] = weights_train.reset_index(drop=True)
    
    test_df = pd.DataFrame(X_test_scaled, columns=['Z_mass', 'Z_pt'])
    test_df['label'] = y_test.reset_index(drop=True)
    test_df['wt'] = weights_test.reset_index(drop=True)
   
   
    # 合并训练集和测试集为最终的DataFrame
    final_chunk = pd.concat([train_df, test_df], ignore_index=True) 
    # 保存处理后的训练集和测试集
    train_df.to_pickle(os.path.join(output_dir, f'train_chunk{chunk_index}.pkl'))
    test_df.to_pickle(os.path.join(output_dir, f'test_chunk{chunk_index}.pkl'))

    # 返回最终合并后的DataFrame
    return final_chunk

output_dir_base = '/vols/cms/xw1523/masterproject'



data_chunk_paths = ['/vols/cms/xw1523/masterproject/data_chunk1.pkl',
                     '/vols/cms/xw1523/masterproject/data_chunk2.pkl', 
                     '/vols/cms/xw1523/masterproject/data_chunk3.pkl', 
                     '/vols/cms/xw1523/masterproject/data_chunk4.pkl', 
                     '/vols/cms/xw1523/masterproject/data_chunk5.pkl']

mc_chunk_paths = ['/vols/cms/xw1523/masterproject/MC_chunk1.pkl', 
                  '/vols/cms/xw1523/masterproject/MC_chunk2.pkl', 
                  '/vols/cms/xw1523/masterproject/MC_chunk3.pkl', 
                  '/vols/cms/xw1523/masterproject/MC_chunk4.pkl', 
                  '/vols/cms/xw1523/masterproject/MC_chunk5.pkl']


for i, (data_path, mc_path) in enumerate(zip(data_chunk_paths, mc_chunk_paths), start=1):
    final_chunk = process_and_save_chunk(data_path, mc_path, output_dir_base, i)
    
    # 保存处理后的数据块
    final_chunk.to_pickle(f'/vols/cms/xw1523/masterproject/final_chunk{i}.pkl')


final_chunk_paths = [
    '/vols/cms/xw1523/masterproject/final_chunk1.pkl',
    '/vols/cms/xw1523/masterproject/final_chunk2.pkl',
    '/vols/cms/xw1523/masterproject/final_chunk3.pkl',
    '/vols/cms/xw1523/masterproject/final_chunk4.pkl',
    '/vols/cms/xw1523/masterproject/final_chunk5.pkl'
]