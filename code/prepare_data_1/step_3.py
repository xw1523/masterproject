import os
import pandas as pd

def merge_chunks(none_dy_dir, real_data_dir, output_dir, chunks=20):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, chunks + 1):
        none_dy_file = f'combined_chunk_{i}.pkl'
        real_data_file = f'combined_chunk_{i}.pkl'
        
        none_dy_path = os.path.join(none_dy_dir, none_dy_file)
        real_data_path = os.path.join(real_data_dir, real_data_file)
        
        if os.path.exists(none_dy_path) and os.path.exists(real_data_path):
            df_none_dy = pd.read_pickle(none_dy_path)
            df_real_data = pd.read_pickle(real_data_path)
            
            # Merge DataFrames
            combined_df = pd.concat([df_none_dy, df_real_data], ignore_index=True)
            # Shuffle the data
            combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True) 
            # Save the merged DataFrame
            output_file_path = os.path.join(output_dir, f'combined_None_DY_Real_Data_chunk_{i}.pkl')
            combined_df.to_pickle(output_file_path)
            print(f'Merged chunk {i} saved to: {output_file_path}')
        else:
            print(f'Chunk {i} not found in one of the directories: {none_dy_dir} or {real_data_dir}')


base_dir = '/vols/cms/yl13923/masterproject/combined_data_chunks'
output_dir = '/vols/cms/yl13923/masterproject/combined_data_chunks/combined_None_DY_Real_Data'
none_dy_dir = os.path.join(base_dir, 'None_DY')
real_data_dir = os.path.join(base_dir, 'Real_Data')

# Merge data chunks
merge_chunks(none_dy_dir, real_data_dir, output_dir)