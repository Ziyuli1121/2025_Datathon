import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering, KMeans, MiniBatchKMeans, Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
from joblib import dump, load
import time
from sklearn.utils import shuffle
import sys
import datetime

warnings.filterwarnings('ignore')
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

INPUT_PATH = 'preprocessed_accounts.csv'
OUTPUT_PATH = 'pytorch_models_optimized'
os.makedirs(OUTPUT_PATH, exist_ok=True)

log_filename = os.path.join(OUTPUT_PATH, f'clustering_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_filename)

print(f"Logging started, output will be saved to: {log_filename}")
print(f"Analysis start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_clustering(X, labels, method_name, sample_size=10000):
    """Calculate multiple clustering evaluation metrics"""
    print(f"\nEvaluating {method_name} clustering quality...")
    metrics = {}
    
    if X.shape[0] > sample_size:
        idx = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[idx]
        labels_sample = labels[idx]
    else:
        X_sample = X
        labels_sample = labels
    
    unique_labels = np.unique(labels_sample)
    n_clusters = len(unique_labels)
    if -1 in unique_labels:  # DBSCAN may include noise points (label -1)
        n_clusters -= 1
        # For DBSCAN, only evaluate non-noise points
        non_noise_idx = labels_sample != -1
        if sum(non_noise_idx) > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(X_sample[non_noise_idx], labels_sample[non_noise_idx])
                metrics['davies_bouldin_score'] = davies_bouldin_score(X_sample[non_noise_idx], labels_sample[non_noise_idx])
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_sample[non_noise_idx], labels_sample[non_noise_idx])
            except Exception as e:
                print(f"Error calculating metrics for non-noise points: {e}")
    else:
        # At least 2 clusters are needed to calculate evaluation metrics
        if n_clusters >= 2:
            try:
                metrics['silhouette_score'] = silhouette_score(X_sample, labels_sample)
                metrics['davies_bouldin_score'] = davies_bouldin_score(X_sample, labels_sample)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_sample, labels_sample)
            except Exception as e:
                print(f"Error calculating metrics: {e}")
    
    print(f"Clustering quality metrics for {method_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    return metrics

def load_and_prepare_data(sample_size=None):
    """Load preprocessed data and prepare for modeling, optional sampling"""
    print("Starting to load preprocessed data...")
    try:
        df = pd.read_csv(INPUT_PATH)
        print(f"Data loaded successfully, original shape: {df.shape}")
        
        if sample_size and sample_size < df.shape[0]:
            df = df.sample(sample_size, random_state=42)
            print(f"Shape after random sampling: {df.shape}")
        
        from clustering_model import create_features, potential_features
        
        # Create derived features
        df_with_features = create_features(df)
        
        # Collect potential features
        all_potential_features = []
        for category, features in potential_features.items():
            all_potential_features.extend(features)
        
        valid_features = [col for col in all_potential_features if col in df_with_features.columns]
        print(f"Valid feature list ({len(valid_features)} features): {valid_features}")
        if len(valid_features) < 3:
            print("Warning: Too few valid features, trying to add more basic features...")
            numeric_cols = df_with_features.select_dtypes(include=['float64', 'int64']).columns.tolist()
            exclude_patterns = ['_nbr', 'id', 'count']
            additional_features = [col for col in numeric_cols if not any(pattern in col.lower() for pattern in exclude_patterns)]
            valid_features.extend([col for col in additional_features if col not in valid_features])
            valid_features = valid_features[:20]  # Limit maximum number of features
        
        X = df_with_features[valid_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        
        print(f"Data preparation complete. Feature matrix shape: {X_scaled.shape}")
        return X_scaled, X, df_with_features, valid_features
    
    except Exception as e:
        print(f"Data preparation error: {e}")
        import traceback
        traceback.print_exc()
        raise

class SmallAutoencoder(nn.Module):
    """Smaller, more efficient PyTorch autoencoder model"""
    def __init__(self, input_dim, encoding_dim=5):
        super(SmallAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, min(64, input_dim)),
            nn.ReLU(),
            nn.Linear(min(64, input_dim), encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, min(64, input_dim)),
            nn.ReLU(),
            nn.Linear(min(64, input_dim), input_dim)
        )
    
    def forward(self, x):
        """Forward propagation function"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

def train_autoencoder(X_scaled, encoding_dim=10, epochs=50, batch_size=128):
    print("\nTraining autoencoder for dimensionality reduction...")
    
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = X_scaled.shape[1]
    model = SmallAutoencoder(input_dim, encoding_dim).to(device)
    print(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'train_loss': [], 'val_loss': []}
    
    val_size = min(int(0.1 * len(X_tensor)), 1000)  # Limit validation set size
    train_tensor, val_tensor = X_tensor[val_size:], X_tensor[:val_size]
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, _ in dataloader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
        
        train_loss = train_loss / len(X_tensor[val_size:])
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            val_loss = criterion(val_outputs, val_tensor).item()
            
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_PATH, 'autoencoder_training.png'))
    plt.close()
    
    model.eval()
    batch_size = 1000
    encoded_features = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            encoded_batch = model.encode(batch).cpu().numpy()
            encoded_features.append(encoded_batch)
    
    encoded_features = np.vstack(encoded_features)
    print(f"Encoded feature shape: {encoded_features.shape}")
    
    model_path = os.path.join(OUTPUT_PATH, 'autoencoder_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Autoencoder model saved to {model_path}")
    
    return encoded_features, model

def apply_birch_clustering(X, n_clusters=4):
    """Apply BIRCH clustering algorithm, suitable for large datasets"""
    print("\nApplying BIRCH clustering algorithm...")
    start_time = time.time()
    
    birch = Birch(n_clusters=n_clusters, threshold=0.5, branching_factor=50)
    labels = birch.fit_predict(X)
    
    metrics = evaluate_clustering(X, labels, "BIRCH")
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Cluster distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    print(f"BIRCH clustering execution time: {time.time()-start_time:.2f} seconds")
    
    return labels

def apply_minibatch_kmeans(X, n_clusters=4):
    """Apply Mini-Batch K-means clustering algorithm, suitable for large datasets"""
    print("\nApplying Mini-Batch K-means clustering algorithm...")
    start_time = time.time()
    
    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                              batch_size=1000, 
                              max_iter=100, 
                              random_state=42)
    labels = mbkmeans.fit_predict(X)
    
    metrics = evaluate_clustering(X, labels, "Mini-Batch K-means")
    
    print(f"Inertia (sum of squared distances to closest centroid): {mbkmeans.inertia_:.4f}")
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Cluster distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    print(f"Mini-Batch K-means clustering execution time: {time.time()-start_time:.2f} seconds")
    
    return labels

def apply_hierarchical_clustering_sampled(X, n_clusters=4, max_samples=10000):
    """Apply hierarchical clustering algorithm, but only process a subset of the data"""
    print("\nApplying hierarchical clustering algorithm (sampled version)...")
    start_time = time.time()
    
    if X.shape[0] > max_samples:
        print(f"Dataset too large ({X.shape[0]} samples), randomly sampling {max_samples} samples for hierarchical clustering...")
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X_sampled = X[indices]
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        sampled_labels = hierarchical.fit_predict(X_sampled)
        
        metrics = evaluate_clustering(X_sampled, sampled_labels, "Hierarchical (sampled)")
        
        unique_labels, counts = np.unique(sampled_labels, return_counts=True)
        print("Sample cluster distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"Cluster {label}: {count} samples ({count/len(sampled_labels)*100:.2f}%)")
        
        print("Using K-means to extend cluster assignments to the entire dataset...")
        
        centroids = np.zeros((n_clusters, X.shape[1]))
        for i in range(n_clusters):
            mask = sampled_labels == i
            if np.sum(mask) > 0:  # Ensure non-empty cluster
                centroids[i] = X_sampled[mask].mean(axis=0)
        
        kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1, max_iter=30)
        labels = kmeans.fit_predict(X)
        
        print("Evaluating extended clustering on full dataset:")
        full_metrics = evaluate_clustering(X, labels, "Hierarchical (full)")
        
    else:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = hierarchical.fit_predict(X)
        
        metrics = evaluate_clustering(X, labels, "Hierarchical")
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Full dataset cluster distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    print(f"Hierarchical clustering execution time: {time.time()-start_time:.2f} seconds")
    
    return labels

def apply_dbscan_optimized(X, eps=0.5, min_samples=5, max_samples=50000):
    """Apply optimized version of DBSCAN clustering algorithm, use sampling for very large datasets"""
    print("\nApplying optimized version of DBSCAN clustering algorithm...")
    start_time = time.time()
    
    if X.shape[0] > max_samples:
        print(f"Dataset too large ({X.shape[0]} samples), randomly sampling {max_samples} samples for parameter optimization...")
        sample_indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X_sample = X[sample_indices]
    else:
        X_sample = X
    
    best_silhouette = -1
    best_eps = eps
    best_min_samples = min_samples
    best_metrics = {}
    
    for eps_val in [0.3, 0.5, 0.7, 1.0]:
        for min_samples_val in [5, 10, 20]:
            sub_sample_size = min(10000, X_sample.shape[0])
            sub_indices = np.random.choice(X_sample.shape[0], sub_sample_size, replace=False)
            X_sub_sample = X_sample[sub_indices]
            
            dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val, n_jobs=1)
            sub_labels = dbscan.fit_predict(X_sub_sample)
            
            n_clusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
            if n_clusters < 2 or n_clusters > 10:
                continue
            
            # Calculate silhouette score (only for non-noise points)
            non_noise_idx = sub_labels != -1
            if sum(non_noise_idx) > 1:
                try:
                    current_metrics = evaluate_clustering(X_sub_sample[non_noise_idx], 
                                                         sub_labels[non_noise_idx], 
                                                         f"DBSCAN (eps={eps_val}, min_samples={min_samples_val})",
                                                         sample_size=sub_sample_size)
                    
                    sil_score = current_metrics.get('silhouette_score', -1)
                    if sil_score > best_silhouette:
                        best_silhouette = sil_score
                        best_eps = eps_val
                        best_min_samples = min_samples_val
                        best_metrics = current_metrics
                except Exception as e:
                    print(f"Error evaluating DBSCAN parameters: {e}")
    
    print(f"Best DBSCAN parameters: eps={best_eps}, min_samples={best_min_samples}")
    print("Best parameter metrics:")
    for metric_name, value in best_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples, n_jobs=1)
    sample_labels = dbscan.fit_predict(X_sample)
    
    if len(set(sample_labels)) > 1:  # Ensure multiple clusters
        sample_metrics = evaluate_clustering(X_sample, sample_labels, "DBSCAN (final sample)")
    
    n_clusters = len(set(sample_labels)) - (1 if -1 in sample_labels else 0)
    
    if X_sample.shape[0] < X.shape[0]:
        print(f"Extending DBSCAN results to the entire dataset ({X.shape[0]} samples)...")
        
        centroids = []
        cluster_labels = []
        
        for i in range(-1, n_clusters):  # Include -1 (noise)
            mask = sample_labels == i
            if np.sum(mask) > 0:
                centroid = X_sample[mask].mean(axis=0)
                centroids.append(centroid)
                cluster_labels.append(i)
        
        centroids = np.array(centroids)
        
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(centroids)
        
        batch_size = 10000
        labels = np.zeros(X.shape[0], dtype=int)
        
        for i in range(0, X.shape[0], batch_size):
            end = min(i + batch_size, X.shape[0])
            batch = X[i:end]
            distances, indices = nn.kneighbors(batch)
            labels[i:end] = np.array(cluster_labels)[indices].flatten()
        
        if len(set(labels)) > 1:  # Ensure multiple clusters
            full_metrics = evaluate_clustering(X, labels, "DBSCAN (full dataset)")
    else:
        labels = sample_labels
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"DBSCAN found {n_clusters} clusters and noise")
    print("Cluster distribution:")
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"Noise: {count} samples ({count/len(labels)*100:.2f}%)")
        else:
            print(f"Cluster {label}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    print(f"DBSCAN execution time: {time.time()-start_time:.2f} seconds")
    
    return labels, n_clusters

def determine_labels(X, cluster_labels, df_with_features, valid_features, method_name):
    """Interpret clustering labels and map to business categories"""
    print(f"\nInterpreting {method_name} clustering results...")
    
    label_names = {
        0: "Can increase credit limit without risk",
        1: "Can increase credit limit with risk",
        2: "No need to increase credit limit",
        3: "High risk account"
    }
    
    df_with_clusters = df_with_features.copy()
    df_with_clusters[f'{method_name}_cluster'] = cluster_labels
    agg_dict = {}
    key_features = [
        'behavior_score_combined', 'external_credit_score', 
        'delinquency_risk', 'cu_nbr_days_dlq', 'credit_utilization',
        'has_fraud'
    ]
    
    for feature in key_features:
        if feature in df_with_clusters.columns:
            agg_dict[feature] = 'mean'
    
    agg_dict[f'{method_name}_cluster'] = 'count'
    
    if len(agg_dict) <= 1:
        for feature in valid_features:
            if feature in df_with_clusters.columns:
                agg_dict[feature] = 'mean'
    
    cluster_analysis = df_with_clusters.groupby(f'{method_name}_cluster').agg(agg_dict)
    cluster_analysis = cluster_analysis.rename(columns={f'{method_name}_cluster': 'count'})
    print("Cluster statistics:")
    print(cluster_analysis)
    
    cluster_scores = {}
    
    positive_features = [
        'behavior_score_combined', 'external_credit_score'
    ]
    
    negative_features = [
        'delinquency_risk', 'cu_nbr_days_dlq', 'has_fraud'
    ]
    
    for cluster_id in cluster_analysis.index:
        total_score = 0
        
        for feature in cluster_analysis.columns:
            if feature == 'count':
                continue
            
            if cluster_analysis[feature].max() > 0:
                normalized_value = cluster_analysis.loc[cluster_id, feature] / cluster_analysis[feature].max()
                
                if feature in positive_features:
                    total_score += normalized_value
                elif feature in negative_features:
                    total_score -= normalized_value
                else:
                    if any(keyword in feature.lower() for keyword in ['score', 'bureau', 'line']):
                        total_score += normalized_value
                    elif any(keyword in feature.lower() for keyword in ['risk', 'dlq', 'fraud', 'due']):
                        total_score -= normalized_value
        
        cluster_scores[cluster_id] = total_score
    
    print("Cluster scores:")
    for cluster_id, score in cluster_scores.items():
        print(f"Cluster {cluster_id}: {score:.4f}")
    
    sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
    
    cluster_mapping = {}
    for i, (cluster_id, _) in enumerate(sorted_clusters):
        if i < 4:  # Map to our 4 categories
            cluster_mapping[cluster_id] = i
        else:
            cluster_mapping[cluster_id] = 3  # Default map to high risk
    
    df_with_clusters[f'{method_name}_segment'] = df_with_clusters[f'{method_name}_cluster'].map(
        lambda x: cluster_mapping.get(x, 3)  # Default to high risk if mapping doesn't exist
    )
    df_with_clusters[f'{method_name}_segment_name'] = df_with_clusters[f'{method_name}_segment'].map(label_names)
    
    output_file = os.path.join(OUTPUT_PATH, f'{method_name}_clustered_accounts.csv')
    df_with_clusters.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return df_with_clusters, cluster_mapping

def create_visualization_sampled(X, labels, method_name, df_with_clusters, max_samples=10000):
    """Use PCA to create 2D visualization of clustering results, use sampling for large datasets"""
    print(f"\nCreating visualization for {method_name} clustering...")
    
    if X.shape[0] > max_samples:
        print(f"Dataset too large ({X.shape[0]} samples), randomly sampling {max_samples} samples for visualization...")
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X_sampled = X[indices]
        labels_sampled = labels[indices]
        segments_sampled = df_with_clusters[f'{method_name}_segment_name'].values[indices]
    else:
        X_sampled = X
        labels_sampled = labels
        segments_sampled = df_with_clusters[f'{method_name}_segment_name'].values
    
    if X_sampled.shape[0] > 10000:
        print("Using Incremental PCA for dimensionality reduction...")
        ipca = IncrementalPCA(n_components=2, batch_size=1000)
        X_pca = ipca.fit_transform(X_sampled)
    else:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_sampled)
    
    viz_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    viz_df[f'{method_name}_cluster'] = labels_sampled
    viz_df[f'{method_name}_segment_name'] = segments_sampled
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=viz_df, 
        x='PC1', 
        y='PC2', 
        hue=f'{method_name}_segment_name', 
        palette='viridis', 
        s=30, 
        alpha=0.6
    )
    plt.title(f'{method_name} Clustering Results (Sampled Data)', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Account Category', fontsize=10)
    
    viz_file = os.path.join(OUTPUT_PATH, f'{method_name}_visualization.png')
    plt.savefig(viz_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {viz_file}")
    
    viz_df.to_csv(os.path.join(OUTPUT_PATH, f'{method_name}_visualization_data.csv'), index=False)

def compare_models(df_original, models_results):
    """Compare results of different clustering methods"""
    print("\nComparing results of different clustering methods...")
    
    comparison_df = pd.DataFrame()
    
    try:
        kmeans_df = pd.read_csv('clustered_accounts.csv')
        if 'segment_name' in kmeans_df.columns:
            comparison_df['KMeans'] = kmeans_df['segment_name']
            models_results['KMeans'] = {
                'df': kmeans_df,
                'segment_col': 'segment_name'
            }
    except Exception as e:
        print(f"Cannot load KMeans results: {e}")
    
    for method_name, result in models_results.items():
        if method_name != 'KMeans':
            comparison_df[method_name] = result['df'][result['segment_col']]
    
    print("Agreement percentage between methods:")
    methods = list(models_results.keys())
    
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method1 = methods[i]
            method2 = methods[j]
            col1 = models_results[method1]['df'][models_results[method1]['segment_col']]
            col2 = models_results[method2]['df'][models_results[method2]['segment_col']]
            agreement = (col1 == col2).mean() * 100
            
            print(f"{method1} vs {method2}: {agreement:.2f}% agreement")
    
    print("\nCategory distribution by method:")
    
    for method_name, result in models_results.items():
        segment_col = result['segment_col']
        counts = result['df'][segment_col].value_counts(normalize=True) * 100
        print(f"\n{method_name} distribution:")
        for category, percentage in counts.items():
            print(f"  {category}: {percentage:.2f}%")
    
    comparison_file = os.path.join(OUTPUT_PATH, 'clustering_methods_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Comparison results saved to {comparison_file}")

def main():
    """Main function to run all clustering methods"""
    print("Starting advanced clustering analysis with memory optimization...")
    start_time = time.time()
    
    try:
        sample_size = None
        X_scaled, X_original, df_with_features, valid_features = load_and_prepare_data(sample_size)
        
        models_results = {}
        
        encoding_dim = min(10, X_scaled.shape[1]//2)
        encoded_features, encoder = train_autoencoder(X_scaled, encoding_dim=encoding_dim, epochs=50, batch_size=128)
        
        mb_kmeans_labels = apply_minibatch_kmeans(encoded_features, n_clusters=4)
        mb_kmeans_df, mb_kmeans_mapping = determine_labels(X_scaled, mb_kmeans_labels, df_with_features, valid_features, 'mbkmeans')
        create_visualization_sampled(encoded_features, mb_kmeans_labels, 'mbkmeans', mb_kmeans_df)
        models_results['MiniBatchKMeans'] = {
            'df': mb_kmeans_df,
            'segment_col': 'mbkmeans_segment_name'
        }
        
        birch_labels = apply_birch_clustering(encoded_features, n_clusters=4)
        birch_df, birch_mapping = determine_labels(X_scaled, birch_labels, df_with_features, valid_features, 'birch')
        create_visualization_sampled(encoded_features, birch_labels, 'birch', birch_df)
        models_results['BIRCH'] = {
            'df': birch_df,
            'segment_col': 'birch_segment_name'
        }
        
        hier_labels = apply_hierarchical_clustering_sampled(encoded_features, n_clusters=4, max_samples=5000)
        hier_df, hier_mapping = determine_labels(X_scaled, hier_labels, df_with_features, valid_features, 'hierarchical')
        create_visualization_sampled(encoded_features, hier_labels, 'hierarchical', hier_df)
        models_results['Hierarchical'] = {
            'df': hier_df,
            'segment_col': 'hierarchical_segment_name'
        }
        
        dbscan_labels, n_clusters = apply_dbscan_optimized(encoded_features, max_samples=10000)
        dbscan_df, dbscan_mapping = determine_labels(X_scaled, dbscan_labels, df_with_features, valid_features, 'dbscan')
        create_visualization_sampled(encoded_features, dbscan_labels, 'dbscan', dbscan_df)
        models_results['DBSCAN'] = {
            'df': dbscan_df,
            'segment_col': 'dbscan_segment_name'
        }
        
        compare_models(df_with_features, models_results)
        
        evaluation_metrics = {}
        
        total_time = time.time()-start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print("Advanced clustering analysis complete!")
        
        print("\n" + "="*80)
        print(f"Analysis end time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"All outputs and results saved to: {OUTPUT_PATH}")
        print(f"Log file path: {log_filename}")
        print("="*80)
        
        if evaluation_metrics:
            metrics_df = pd.DataFrame(evaluation_metrics)
            metrics_file = os.path.join(OUTPUT_PATH, 'clustering_evaluation_metrics.csv')
            metrics_df.to_csv(metrics_file)
            print(f"Clustering evaluation metrics saved to {metrics_file}")
        
    except Exception as e:
        print(f"Advanced clustering analysis error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*80)
        print(f"Analysis terminated due to error: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file path: {log_filename}")
        print("="*80)

if __name__ == "__main__":
    main()
