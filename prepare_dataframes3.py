import os
import pandas as pd

mc_chunk_list = [
    '/vols/cms/xw1523/masterproject/MC_chunks/MC_chunk_0.pkl',
    '/vols/cms/xw1523/masterproject/MC_chunks/MC_chunk_1.pkl',
    '/vols/cms/xw1523/masterproject/MC_chunks/MC_chunk_2.pkl',
    '/vols/cms/xw1523/masterproject/MC_chunks/MC_chunk_3.pkl',
    '/vols/cms/xw1523/masterproject/MC_chunks/MC_chunk_4.pkl'
]

combine_chunk_list = [
    '/vols/cms/xw1523/masterproject/combined_chunks/combined_chunk_0.pkl',
    '/vols/cms/xw1523/masterproject/combined_chunks/combined_chunk_1.pkl',
    '/vols/cms/xw1523/masterproject/combined_chunks/combined_chunk_2.pkl',
    '/vols/cms/xw1523/masterproject/combined_chunks/combined_chunk_3.pkl',
    '/vols/cms/xw1523/masterproject/combined_chunks/combined_chunk_4.pkl'
]

def add_label_and_merge(mc_chunk_list, combine_chunk_list):
    # 循环处理每对MC和Combine chunk
    for i, (mc_path, combine_path) in enumerate(zip(mc_chunk_list, combine_chunk_list)):
        # 读取MC和Combine chunk数据
        mc_df = pd.read_pickle(mc_path)
        combine_df = pd.read_pickle(combine_path)
        
        # 为MC和Combine chunk添加标签列
        mc_df['label'] = 0
        combine_df['label'] = 1
        
        # 合并MC和Combine chunk
        merged_dff = pd.concat([mc_df, combine_df], ignore_index=True)
        
        # 洗牌（shuffle）合并后的DataFrame
        merged_dff = merged_dff.sample(frac=1).reset_index(drop=True)
        
        # 保存合并后的DataFrame到final_chunks目录下的对应文件中
        final_chunk_path = f'/vols/cms/xw1523/masterproject/final_chunks/final_chunk_{i}.pkl'
        merged_dff.to_pickle(final_chunk_path)
        
        # 打印合并后的DataFrame的头部
        print(f"Merged and shuffled DataFrame for final_chunk_{i}:")
        print(merged_dff.head())
        
# 调用函数进行处理
add_label_and_merge(mc_chunk_list, combine_chunk_list)
