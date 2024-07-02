import uproot3
import pandas as pd
import json
import os
import gc

def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def process_files_to_dataframe(file_path, json_data, lum=59830, is_dy=False, adjust_weight=True):
    branches = ['Z_mass', 'Z_pt', 'n_jets', 'n_deepbjets', 'mjj', 'jdeta', 'jdphi', 'dijetpt', 'jpt_1', 'jpt_2', 'jpt_3', 'wt','U1','U2','met','met_phi']

    json_file_name = os.path.basename(file_path).split('_zmm_2018')[0]
    xs = evt = 1  # Default values for real data
    if adjust_weight and json_file_name in json_data: 
        xs = json_data[json_file_name]['xs']
        evt = json_data[json_file_name]['evt']

    with uproot3.open(file_path) as file:
        df = file['ntuple'].pandas.df(branches)
        if adjust_weight:  
            df['wt'] *= xs * lum / evt
            if not is_dy:
                df['wt'] = -df['wt']
    
    return df

json_data = read_json('/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/params_UL_2018.json')

None_DY_list = [
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/Tbar-tW_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/WZTo1L3Nu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/T-tW_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/W4JetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/W1JetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/EWKZ2Jets_ZToLL_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/EWKWMinus2Jets_WToLNu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/WGToLNuG_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/EWKWPlus2Jets_WToLNu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/WWTo2L2Nu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/WZTo3LNu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/WZTo1L1Nu2Q_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/WJetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/Tbar-t_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/WWTo1L1Nu2Q_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/W3JetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/WZTo2Q2L_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/W2JetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/ZZTo4L_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/T-t_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/ZZTo2L2Nu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/TTTo2L2Nu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/TTToHadronic_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/TTToSemiLeptonic_zmm_2018.root"]


DY_list = [
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DY2JetsToLL-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DYJetsToLL_M-10to50-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DY3JetsToLL-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DY4JetsToLL-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DYJetsToLL-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DYJetsToLL-LO-ext1_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/DY1JetsToLL-LO_zmm_2018.root'
]


Real_Data_list = [
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/SingleMuonC_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/SingleMuonA_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/SingleMuonB_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v4/SingleMuonD_zmm_2018.root',
]

output_dir_base = '/vols/cms/yl13923/masterproject/data_chunks'

# Process and save each category
for category, file_paths in [('None_DY', None_DY_list), ('DY', DY_list), ('Real_Data', Real_Data_list)]:
    adjust_weight = category != 'Real_Data'
    for file_path in file_paths:
        df = process_files_to_dataframe(file_path, json_data, is_dy=(category == 'DY'), adjust_weight=adjust_weight)
        output_dir = os.path.join(output_dir_base, category)
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.basename(file_path).replace('.root', '.pkl')
        output_path = os.path.join(output_dir, file_name)
        df.to_pickle(output_path)
        print(f"Processed file saved to: {output_path}")

print("All files processed and saved successfully.")

# # Since it was killed when there was only one file left, the second time only the last file of real data (SingleMuonD) is run.
# category = 'Real_Data'
# adjust_weight = False  # Real_Data
# for file_path in Real_Data_list:
#     df = process_files_to_dataframe(file_path, json_data, is_dy=False, adjust_weight=adjust_weight)
#     output_dir = os.path.join(output_dir_base, category)
#     os.makedirs(output_dir, exist_ok=True)
#     file_name = os.path.basename(file_path).replace('.root', '.pkl')
#     output_path = os.path.join(output_dir, file_name)
#     df.to_pickle(output_path)
#     print(f"Processed file saved to: {output_path}")

# print("Real_Data files processed and saved successfully.")