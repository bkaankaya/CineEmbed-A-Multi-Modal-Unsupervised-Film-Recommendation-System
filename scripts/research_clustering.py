
import os
import sys
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from cineembed.data import load_feature_matrix, get_labels, get_block_indices
from cineembed.eval import (
    cluster_assignments_kmeans, 
    cluster_assignments_gmm, 
    cluster_assignments_hdbscan,
    multilabel_macro_nmi
)

def main():
    print("Loading data...")
    X, feature_names = load_feature_matrix('artifacts/feature_matrix.npz')
    labels_dict = get_labels('artifacts/movies_eda_final.csv')
    block_indices = get_block_indices(feature_names)
    
    # Take a 10k subset for speed
    n_samples = 10000
    indices = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
    X_sub = X[indices].numpy()
    
    # We need a latent space to test on. Since we don't have all models loaded,
    # let's use the genre block itself as a "perfect" latent for genre clustering
    # to see how metrics behave, and maybe a random projection for a "noisy" latent.
    genre_block = X_sub[:, block_indices['genre']]
    
    # Also get a "latent" z by random projection to 64d (to simulate our model's output)
    z_rand = X_sub @ np.random.RandomState(42).randn(X_sub.shape[1], 64)
    
    # Ground truth for genre
    y_genre = labels_dict['primary_genre'][indices]
    le = LabelEncoder()
    y_genre_int = le.fit_transform(y_genre)
    
    print(f"\n--- GMM Stability (z_rand, k=21, 5 seeds) ---")
    nmis = []
    amis = []
    for seed in range(40, 45):
        c = cluster_assignments_gmm(z_rand, k=21, seed=seed)
        nmi = normalized_mutual_info_score(y_genre_int, c)
        ami = adjusted_mutual_info_score(y_genre_int, c)
        nmis.append(nmi)
        amis.append(ami)
        print(f"Seed {seed}: NMI={nmi:.4f}, AMI={ami:.4f}")
    print(f"GMM NMI mean: {np.mean(nmis):.4f}, std: {np.std(nmis):.4f}")
    
    print(f"\n--- HDBSCAN Noise Handling (z_rand) ---")
    c_hdb = cluster_assignments_hdbscan(z_rand, min_cluster_size=50)
    n_noise = (c_hdb == -1).sum()
    print(f"HDBSCAN found {len(np.unique(c_hdb)) - (1 if -1 in c_hdb else 0)} clusters")
    print(f"Noise points (-1): {n_noise} ({n_noise/n_samples:.1%})")
    
    # Scenario A: -1 is one cluster
    nmi_a = normalized_mutual_info_score(y_genre_int, c_hdb)
    ami_a = adjusted_mutual_info_score(y_genre_int, c_hdb)
    print(f"Scenario A (Noise as one cluster): NMI={nmi_a:.4f}, AMI={ami_a:.4f}")
    
    # Scenario B: -1 points are unique clusters (maximum penalty)
    c_hdb_b = c_hdb.copy()
    noise_mask = (c_hdb == -1)
    c_hdb_b[noise_mask] = np.arange(max(c_hdb)+1, max(c_hdb)+1+n_noise)
    nmi_b = normalized_mutual_info_score(y_genre_int, c_hdb_b)
    ami_b = adjusted_mutual_info_score(y_genre_int, c_hdb_b)
    print(f"Scenario B (Each noise point unique): NMI={nmi_b:.4f}, AMI={ami_b:.4f}")

    # Scenario C: Exclude noise from evaluation
    keep_mask = (c_hdb != -1)
    if keep_mask.any():
        nmi_c = normalized_mutual_info_score(y_genre_int[keep_mask], c_hdb[keep_mask])
        ami_c = adjusted_mutual_info_score(y_genre_int[keep_mask], c_hdb[keep_mask])
        print(f"Scenario C (Exclude noise): NMI={nmi_c:.4f}, AMI={ami_c:.4f}")
    
    print(f"\n--- AMI vs NMI Bias (Genre is imbalanced) ---")
    # Genre distribution
    counts = pd.Series(y_genre).value_counts()
    print(f"Genre imbalance: top category {counts.index[0]} has {counts.iloc[0]} samples ({counts.iloc[0]/n_samples:.1%})")
    
    print(f"\n--- Multi-label Macro NMI ---")
    # Use the actual genre block (excluding has_genre if it's there)
    # genre block prefix rules say 'genre_' and 'has_genre'
    genre_names = [feature_names[i] for i in range(block_indices['genre'].start, block_indices['genre'].stop)]
    genre_cols = [i for i, name in enumerate(genre_names) if name != 'has_genre']
    genre_onehot = genre_block[:, genre_cols]
    
    res_macro = multilabel_macro_nmi(c_hdb, genre_onehot, metric='nmi')
    print(f"HDBSCAN Macro NMI: {res_macro['macro']:.4f} (evaluated {res_macro['n_genres_evaluated']} genres)")

if __name__ == "__main__":
    main()
