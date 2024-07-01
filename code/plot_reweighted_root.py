import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import ROOT
import joblib
import os
import itertools
import shutil
# Load the model
model = tf.keras.models.load_model('/vols/cms/yl13923/masterproject/my_model.h5')

# Paths for files and scalers
file_scaler_map = {
    '/vols/cms/yl13923/masterproject/final_chunks/final_chunks/final_chunk_1.pkl': '/vols/cms/yl13923/masterproject/scaler/scaler_1.joblib',
    '/vols/cms/yl13923/masterproject/final_chunks/final_chunks/final_chunk_2.pkl': '/vols/cms/yl13923/masterproject/scaler/scaler_2.joblib',
}

feature_names = ['Z_mass', 'Z_pt', 'n_jets', 'n_deepbjets', 'mjj', 'jdeta', 'jdphi', 'dijetpt', 'jpt_1', 'jpt_2', 'jpt_3']

batch_size = 500000

# Create a temporary directory to store intermediate results
temp_dir = '/vols/cms/yl13923/masterproject/temp_data_1/'
os.makedirs(temp_dir, exist_ok=True)

# Generator function
def get_generator(file_scaler_map, batch_size=550000):
    def generator():
        for file_path, scaler_path in file_scaler_map.items():
            df = pd.read_pickle(file_path)
            X = df[feature_names].values
            y = df['label'].values
            weights = df['wt'].values
            
            scaler = joblib.load(scaler_path)
            X_original = scaler.inverse_transform(X)
            
            num_samples = len(df)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                yield (X[start:end], y[start:end], weights[start:end], X_original[start:end], file_path)
    return generator

# Initialize storage
data_stats = {name: {'mc': [], 'mc_weights': [], 'reweighted_mc': [], 'reweighted_mc_weights': [], 'real_data': [], 'real_data_weights': []} for name in feature_names}

# Process data using the generator
data_generator = get_generator(file_scaler_map, batch_size=batch_size)


# Process data and save intermediate results
for X_batch, y_batch, weights_batch, X_original_batch, file_path in data_generator():
    predictions = model.predict(X_batch).ravel()
    
    f_x = np.minimum(predictions, 0.99999)
    reweight = f_x / (1.0 - f_x)
    new_weights_batch = reweight * weights_batch
    
    temp_data = {name: {'mc': [], 'mc_weights': [], 'reweighted_mc': [], 'reweighted_mc_weights': [], 'real_data': [], 'real_data_weights': []} for name in feature_names}

    for i in range(len(X_batch)):
        X = X_batch[i]
        y = y_batch[i]
        weight = weights_batch[i]
        new_weight = new_weights_batch[i]
        X_original = X_original_batch[i]
        
        for j, name in enumerate(feature_names):
            if y == 0:  # MC
                temp_data[name]['mc'].append(X_original[j])
                temp_data[name]['mc_weights'].append(weight)
                temp_data[name]['reweighted_mc'].append(X_original[j])
                temp_data[name]['reweighted_mc_weights'].append(new_weight)
            elif y == 1:  # Real data
                temp_data[name]['real_data'].append(X_original[j])
                temp_data[name]['real_data_weights'].append(weight)
    
    # Save intermediate results to temporary file
    temp_file = os.path.join(temp_dir, os.path.basename(file_path) + '.npz')
    np.savez(temp_file, **{f"{name}_{key}": temp_data[name][key] for name in feature_names for key in temp_data[name]})

# Aggregate all intermediate results
for temp_file in os.listdir(temp_dir):
    temp_data = np.load(os.path.join(temp_dir, temp_file))
    for name in feature_names:
        data_stats[name]['mc'].extend(temp_data[f"{name}_mc"])
        data_stats[name]['mc_weights'].extend(temp_data[f"{name}_mc_weights"])
        data_stats[name]['reweighted_mc'].extend(temp_data[f"{name}_reweighted_mc"])
        data_stats[name]['reweighted_mc_weights'].extend(temp_data[f"{name}_reweighted_mc_weights"])
        data_stats[name]['real_data'].extend(temp_data[f"{name}_real_data"])
        data_stats[name]['real_data_weights'].extend(temp_data[f"{name}_real_data_weights"])

chi2_1D_results = {}

# Plot 1D ratio graphs
for name in feature_names:
    all_values = np.array(data_stats[name]['mc'] + data_stats[name]['reweighted_mc'] + data_stats[name]['real_data'])
    mean = np.mean(all_values)
    std = np.std(all_values)
    range_min = mean - std
    range_max = mean + 11 * std
 
    # Convert numpy histograms to ROOT histograms
    h_mc = ROOT.TH1F(f"h_mc_{name}", f"MC Distribution for {name}", 50, range_min, range_max)
    h_reweighted_mc = ROOT.TH1F(f"h_reweighted_mc_{name}", f"Reweighted MC Distribution for {name}", 50, range_min, range_max)
    h_real_data = ROOT.TH1F(f"h_real_data_{name}", f"Real Data Distribution for {name}", 50, range_min, range_max)
    
    # Fill ROOT histograms
    for val, weight in zip(data_stats[name]['mc'], data_stats[name]['mc_weights']):
        h_mc.Fill(val, weight)
    for val, weight in zip(data_stats[name]['reweighted_mc'], data_stats[name]['reweighted_mc_weights']):
        h_reweighted_mc.Fill(val, weight)
    for val, weight in zip(data_stats[name]['real_data'], data_stats[name]['real_data_weights']):
        h_real_data.Fill(val, weight)
  
    # Perform chi-squared tests using Chi2Test to obtain chi-square statistic and p-value separately.
    chi2_stat_mc_real = h_mc.Chi2Test(h_real_data, "WW CHI2")  # Get chi-square statistic
    p_value_mc_real = h_mc.Chi2Test(h_real_data, "WW")         # Get p-value

    chi2_stat_reweighted_mc_real = h_reweighted_mc.Chi2Test(h_real_data, "WW CHI2")  # Get chi-square statistic
    p_value_reweighted_mc_real = h_reweighted_mc.Chi2Test(h_real_data, "WW")         # Get p-value

    # Store the results
    chi2_1D_results[name] = {
        "MC vs Real Data": {"Chi2": chi2_stat_mc_real, "p_value": p_value_mc_real},
        "Reweighted MC vs Real Data": {"Chi2": chi2_stat_reweighted_mc_real, "p_value": p_value_reweighted_mc_real}
    }
    
    # create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot distributions
    ax1.hist(data_stats[name]['mc'], bins=50, alpha=0.5, label='MC', weights=data_stats[name]['mc_weights'], range=(range_min, range_max), histtype='step', color='blue')
    ax1.hist(data_stats[name]['reweighted_mc'], bins=50, alpha=0.5, label='Reweighted MC', weights=data_stats[name]['reweighted_mc_weights'], range=(range_min, range_max), histtype='step', color='green')
    ax1.hist(data_stats[name]['real_data'], bins=50, alpha=0.5, label='Real Data', weights=data_stats[name]['real_data_weights'], range=(range_min, range_max), histtype='step', color='red')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of {name}')
    ax1.legend()
    
    # Calculate ratios
    bins = np.linspace(range_min, range_max, 51)
    mc_hist, _ = np.histogram(data_stats[name]['mc'], bins=bins, weights=data_stats[name]['mc_weights'])
    reweighted_mc_hist, _ = np.histogram(data_stats[name]['reweighted_mc'], bins=bins, weights=data_stats[name]['reweighted_mc_weights'])
    real_data_hist, _ = np.histogram(data_stats[name]['real_data'], bins=bins, weights=data_stats[name]['real_data_weights'])
    
    ratio_mc = mc_hist / (real_data_hist + 1e-6)
    ratio_reweighted_mc = reweighted_mc_hist / (real_data_hist + 1e-6)
    
    # Plot ratio plot
    ax2.plot(bins[:-1], ratio_mc, drawstyle='steps-post', label='MC / Real Data', color='blue')
    ax2.plot(bins[:-1], ratio_reweighted_mc, drawstyle='steps-post', label='Reweighted MC / Real Data', color='green')
    ax2.axhline(1, color='gray', linestyle='--', label='Ratio = 1')
    ax2.set_xlabel(name)
    ax2.set_ylabel('Ratio')
    ax2.set_ylim(0, 2)
    ax2.legend()

    plt.savefig(f'/vols/cms/yl13923/masterproject/plots_2/1D_7/{name}_weighted_distribution_with_ratio.png')
    plt.close()

# 创建一个空的 DataFrame，用于存储结果
chi2_summary_1D_df = pd.DataFrame({
    'Feature': [],
    'Chi2 (before)': [],
    'p-value (before)': [],
    'Chi2 (after)': [],
    'p-value (after)': []
})

# Fill the DataFrame with results
for name, results in chi2_1D_results.items():
    new_row = pd.DataFrame({
        'Feature': [name],
        'Chi2 (before)': [results["MC vs Real Data"]["Chi2"]],
        'p-value (before)': [results["MC vs Real Data"]["p_value"]],
        'Chi2 (after)': [results["Reweighted MC vs Real Data"]["Chi2"]],
        'p-value (after)': [results["Reweighted MC vs Real Data"]["p_value"]]
    })
    chi2_summary_1D_df = pd.concat([chi2_summary_1D_df, new_row], ignore_index=True)

# Print formatted chi-squared test results
print("Chi-squared test results for 1D histograms:")
print(chi2_summary_1D_df)

chi2_2D_results = {}

# 2D
for name1, name2 in itertools.combinations(feature_names, 2):
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Define feature data
    mc_data_x = np.array(data_stats[name1]['mc'])
    mc_data_y = np.array(data_stats[name2]['mc'])
    reweighted_mc_data_x = np.array(data_stats[name1]['reweighted_mc'])
    reweighted_mc_data_y = np.array(data_stats[name2]['reweighted_mc'])
    real_data_x = np.array(data_stats[name1]['real_data'])
    real_data_y = np.array(data_stats[name2]['real_data'])
    
    mc_weights = np.array(data_stats[name1]['mc_weights'])
    reweighted_mc_weights = np.array(data_stats[name1]['reweighted_mc_weights'])
    real_data_weights = np.array(data_stats[name1]['real_data_weights'])

    # Set data ranges
    mean_x = np.mean(mc_data_x)
    std_x = np.std(mc_data_x)
    range_min_x = mean_x -  std_x
    range_max_x = mean_x + 11 * std_x

    mean_y = np.mean(mc_data_y)
    std_y = np.std(mc_data_y)
    range_min_y = mean_y - std_y
    range_max_y = mean_y + 11 * std_y
    
    bins = 10
    xedges = np.linspace(range_min_x, range_max_x, bins + 1)
    yedges = np.linspace(range_min_y, range_max_y, bins + 1)

    nx = len(xedges) - 1
    ny = len(yedges) - 1
    #初始化直方图
    h2_mc = ROOT.TH2F("h2_mc", "h2_mc", nx, xedges[0], xedges[-1], ny, yedges[0], yedges[-1])
    h2_reweighted_mc = ROOT.TH2F("h2_reweighted_mc", "h2_reweighted_mc", nx, xedges[0], xedges[-1], ny, yedges[0], yedges[-1])
    h2_real = ROOT.TH2F("h2_real", "h2_real", nx, xedges[0], xedges[-1], ny, yedges[0], yedges[-1])

    # 填充直方图
    for x, y, weight in zip(mc_data_x, mc_data_y, mc_weights):
        h2_mc.Fill(x, y, weight)
    for x, y, weight in zip(reweighted_mc_data_x, reweighted_mc_data_y, reweighted_mc_weights):
        h2_reweighted_mc.Fill(x, y, weight)
    for x, y, weight in zip(real_data_x, real_data_y, real_data_weights):
        h2_real.Fill(x, y, weight)


# Perform chi-squared tests to obtain chi-square statistic and p-value separately.
    chi2_stat_mc_real = h2_mc.Chi2Test(h2_real, "WW CHI2")  # Get chi-square statistic
    p_value_mc_real = h2_mc.Chi2Test(h2_real, "WW")         # Get p-value

    chi2_stat_reweighted_mc_real = h2_reweighted_mc.Chi2Test(h2_real, "WW CHI2")  # Get chi-square statistic
    p_value_reweighted_mc_real = h2_reweighted_mc.Chi2Test(h2_real, "WW")         # Get p-value

    # Store the results
    chi2_2D_results[(name1, name2)] = {
        "MC vs Real Data": {"Chi2": chi2_stat_mc_real, "p_value": p_value_mc_real},
        "Reweighted MC vs Real Data": {"Chi2": chi2_stat_reweighted_mc_real, "p_value": p_value_reweighted_mc_real}
    }

    # Calculate ratio 
    H_mc, xedges, yedges = np.histogram2d(mc_data_x, mc_data_y, bins=10, range=[[range_min_x, range_max_x], [range_min_y, range_max_y]], weights=mc_weights)
    H_reweighted_mc, _, _ = np.histogram2d(reweighted_mc_data_x, reweighted_mc_data_y, bins=[xedges, yedges], weights=reweighted_mc_weights)
    H_real, _, _ = np.histogram2d(real_data_x, real_data_y, bins=[xedges, yedges], weights=real_data_weights)
    ratio_mc = H_mc / (H_real + 1e-6) 
    ratio_reweighted_mc = H_reweighted_mc / (H_real + 1e-6)

    # Plot MC / Real Data
    im = axs[0].imshow(ratio_mc.T, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='bwr', vmin=0, vmax=2)
    axs[0].set_title('2D Histogram of MC/Real Data')
    axs[0].set_xlabel(name1)
    axs[0].set_ylabel(name2)
    fig.colorbar(im, ax=axs[0])

    # Plot Reweighted MC / Real Data
    im2 = axs[1].imshow(ratio_reweighted_mc.T, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='bwr', vmin=0, vmax=2)
    axs[1].set_title('2D Histogram of Reweighted MC/Real Data')
    axs[1].set_xlabel(name1)
    axs[1].set_ylabel(name2)
    fig.colorbar(im2, ax=axs[1])

    # Ensure output directory exists
    output_directory = '/vols/cms/yl13923/masterproject/plots_2/2D_7'
    os.makedirs(output_directory, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(output_directory, f'{name1}_vs_{name2}_2d_histograms.png'))
    plt.close()

# 首先定义一个空的DataFrame，明确列的命名
chi2_summary_2D_df = pd.DataFrame(columns=[
    'Feature Pair',
    'Chi2 (before)',
    'p-value (before)',
    'Chi2 (after)',
    'p-value (after)'
])

# Prepare a list to collect row data
rows = []
for (name1, name2), results in chi2_2D_results.items():
    row = {
        'Feature Pair': f"{name1} vs {name2}",
        'Chi2 (before)': results["MC vs Real Data"]["Chi2"],
        'p-value (before)': results["MC vs Real Data"]["p_value"],
        'Chi2 (after)': results["Reweighted MC vs Real Data"]["Chi2"],
        'p-value (after)': results["Reweighted MC vs Real Data"]["p_value"]
    }
    rows.append(row)  # Here we use Python list's append method

# Create DataFrame from the list of dictionaries
chi2_summary_2D_df = pd.DataFrame(rows)
print("Chi-squared test results for 2D histograms:")
print(chi2_summary_2D_df)

# Clean up the temporary directory
shutil.rmtree(temp_dir)