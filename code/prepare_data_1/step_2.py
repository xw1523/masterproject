import os
import pandas as pd

def split_and_combine_chunks(input_dir, output_dir, chunks=20):
    all_files = sorted(os.listdir(input_dir))  
    chunks_data = [[] for _ in range(chunks)] 
    # Process each file, splitting it into chunks
    for file_name in all_files:
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_pickle(file_path)

        # Randomly shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        num_rows = len(df)
        rows_per_chunk = max(num_rows // chunks, 1)  # Calculate the number of rows per chunk

        # Split the DataFrame into the corresponding chunks
        for i in range(chunks):
            start_idx = i * rows_per_chunk
            end_idx = min((i + 1) * rows_per_chunk, num_rows)
            if start_idx < num_rows:
                chunks_data[i].append(df.iloc[start_idx:end_idx])
        del df

    # Merge chunks with the same number and save them
    os.makedirs(output_dir, exist_ok=True) 
    for i, chunk_list in enumerate(chunks_data):
        combined_chunk = pd.concat(chunk_list, ignore_index=True)
        chunk_file_path = os.path.join(output_dir, f'combined_chunk_{i+1}.pkl')
        combined_chunk.to_pickle(chunk_file_path)
        print(f'Combined chunk {i+1} saved to: {chunk_file_path}')

        del combined_chunk
        del chunk_list

base_dir = '/vols/cms/yl13923/masterproject/data_chunks'
output_dir_base = '/vols/cms/yl13923/masterproject/combined_data_chunks'

# Sometimes it works, but sometimes it killed and need to works one by one
categories = ['None_DY', 'DY', 'Real_Data']
# categories = ['None_DY']
# categories = ['DY']
# categories = ['Real_Data']

for category in categories:
    input_dir = os.path.join(base_dir, category)
    output_dir = os.path.join(output_dir_base, category)
    print(f'Processing and combining chunks for category: {category}')
    split_and_combine_chunks(input_dir, output_dir)