import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os
import itertools
import shutil
import matplotlib.ticker as mticker


# Load the model
model = tf.keras.models.load_model('/vols/cms/yl13923/masterproject/my_model.h5')

# Paths for files and scalers
file_scaler_map = {
    '/vols/cms/yl13923/masterproject/final_chunks/final_chunks/final_chunk_1.pkl': '/vols/cms/yl13923/masterproject/scaler/scaler_1.joblib',
    '/vols/cms/yl13923/masterproject/final_chunks/final_chunks/final_chunk_2.pkl': '/vols/cms/yl13923/masterproject/scaler/scaler_2.joblib',
}

feature_names = ['Z_mass', 'Z_pt', 'n_jets', 'n_deepbjets', 'mjj', 'jdeta', 'jdphi', 'dijetpt', 'jpt_1', 'jpt_2', 'jpt_3']

# Set the batch size
batch_size = 500000
# Create a temporary directory to store intermediate results
temp_dir = '/vols/cms/yl13923/masterproject/temp_data_1/'
os.makedirs(temp_dir, exist_ok=True)

# Generator function
def get_generator(file_scaler_map, batch_size=500000):
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

# Calculate the range for each feature and plot
for name in feature_names:
    
    all_values = np.array(data_stats[name]['mc'] + data_stats[name]['reweighted_mc'] + data_stats[name]['real_data'])
    mean = np.mean(all_values)
    std = np.std(all_values)
    range_min = mean - std
    range_max = mean + 11 * std
    
    bins = np.linspace(range_min, range_max, 51)
    real_data_hist, _ = np.histogram(data_stats[name]['real_data'], bins=bins, weights=data_stats[name]['real_data_weights'])
    mc_hist, _ = np.histogram(data_stats[name]['mc'], bins=bins, weights=data_stats[name]['mc_weights'])
    reweighted_mc_hist, _ = np.histogram(data_stats[name]['reweighted_mc'], bins=bins, weights=data_stats[name]['reweighted_mc_weights'])
    
    epsilon = 1e-6  
    real_errors = np.sqrt(real_data_hist) + epsilon
    mc_errors = np.sqrt(mc_hist) + epsilon
    total_errors = np.sqrt(real_errors**2 + mc_errors**2)

    pulls = (real_data_hist - mc_hist) / total_errors # Calculate pull values

   # Create subplots
    fig, (ax1, ax_pull) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot the histograms on the top subplot
    ax1.hist(data_stats[name]['mc'], bins=50, alpha=0.5, label='MC', weights=data_stats[name]['mc_weights'], range=(range_min, range_max), histtype='step', color='blue')
    ax1.hist(data_stats[name]['reweighted_mc'], bins=50, alpha=0.5, label='Reweighted MC', weights=data_stats[name]['reweighted_mc_weights'], range=(range_min, range_max), histtype='step', color='green')
    ax1.hist(data_stats[name]['real_data'], bins=50, alpha=0.5, label='Real Data', weights=data_stats[name]['real_data_weights'], range=(range_min, range_max), histtype='step', color='red')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of {name}')
    ax1.legend()
    

    # Pull plot - using scatter and errorbar
    mid_points = (bins[:-1] + bins[1:]) / 2
    ax_pull.errorbar(mid_points, pulls, yerr=1, fmt='o', color='black', label='Pull value')
    ax_pull.axhline(0, color='black', linestyle='dashed')
    ax_pull.set_ylim([-5, 5])  # Set y-axis limits
    ax_pull.set_ylabel('Pull')
    ax_pull.set_xlabel(name)

    # Set Pull plot background color bands
    ax_pull.fill_between(mid_points, -1, 1, color='green', alpha=0.3)
    ax_pull.fill_between(mid_points, -2, 2, color='yellow', alpha=0.3)

    # Add ticks and labels
    ax_pull.xaxis.set_major_locator(mticker.FixedLocator(mid_points))
    ax_pull.set_xticklabels([f'{int(mp)}' for mp in mid_points], rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(f'/vols/cms/yl13923/masterproject/plots_2/1D_pull/{name}_distribution_with_pull.png')
    plt.close()

# Iterate through all feature combinations to plot 2D histograms and pull plots
for name1, name2 in itertools.combinations(feature_names, 2):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

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

    # Get data ranges
    mean_x = np.mean(mc_data_x)
    std_x = np.std(mc_data_x)
    range_min_x = mean_x - 1 * std_x
    range_max_x = mean_x + 7 * std_x

    mean_y = np.mean(mc_data_y)
    std_y = np.std(mc_data_y)
    range_min_y = mean_y - 1 * std_y
    range_max_y = mean_y + 11 * std_y
    
    bins = 10
    xedges = np.linspace(range_min_x, range_max_x, bins + 1)
    yedges = np.linspace(range_min_y, range_max_y, bins + 1)


    # Compute histograms
    H_mc, xedges, yedges = np.histogram2d(mc_data_x, mc_data_y, bins=[xedges, yedges], weights=mc_weights)
    H_reweighted_mc, _, _ = np.histogram2d(reweighted_mc_data_x, reweighted_mc_data_y, bins=[xedges, yedges], weights=reweighted_mc_weights)
    H_real, _, _ = np.histogram2d(real_data_x, real_data_y, bins=[xedges, yedges], weights=real_data_weights)

    ratio_mc = H_mc / (H_real + 1e-6)   # Avoid division by zero
    ratio_reweighted_mc = H_reweighted_mc / (H_real + 1e-6)

    # Plot MC / Real data ratio
    im = axs[0].imshow(ratio_mc.T, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='bwr', vmin=0, vmax=2)
    axs[0].set_title('2D Histogram of MC/Real Data')
    axs[0].set_xlabel(name1)
    axs[0].set_ylabel(name2)
    fig.colorbar(im, ax=axs[0])

    # Plot Reweighted MC / Real data ratio
    im2 = axs[1].imshow(ratio_reweighted_mc.T, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='bwr', vmin=0, vmax=2)
    axs[1].set_title('2D Histogram of Reweighted MC/Real Data')
    axs[1].set_xlabel(name1)
    axs[1].set_ylabel(name2)
    fig.colorbar(im2, ax=axs[1])
   
    # Calculate errors and pull values
    epsilon = 1e-6 
    mc_errors = np.sqrt(H_mc) + epsilon
    real_errors = np.sqrt(H_real) + epsilon
    total_errors = np.sqrt(real_errors**2 + mc_errors**2)
    pulls = (H_real - H_mc) / total_errors

    # Plot Pull Plot
    im3 = axs[2].imshow(pulls.T, origin='lower', cmap='bwr', aspect='auto', vmin=-5, vmax=5, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    fig.colorbar(im2, ax=axs[2])
    axs[2].set_title('Pull Plot')
    axs[2].set_xlabel(name1)
    axs[2].set_ylabel(name2)

    # Save
    plt.tight_layout()
    plt.savefig(f'/vols/cms/yl13923/masterproject/plots_2/2D_pull/{name1}_vs_{name2}_pull_plots.png')
    plt.close()

# Clean up the temporary directory
shutil.rmtree(temp_dir)