import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
os.environ["LOKY_MAX_CPU_COUNT"] = "2"  # Limit to 2 cores to reduce resource requirements

INPUT_PATH = 'preprocessed_accounts.csv'
OUTPUT_PATH = '.'
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Starting to load preprocessed data...")
# Load preprocessed data
try:
    df = pd.read_csv(INPUT_PATH)
    # View basic data information
    print(f"Data loaded successfully, shape: {df.shape}")
    print(f"Data columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Failed to load data: {e}")
    exit(1)

# Feature selection and engineering - Create credit risk and spending behavior related features
def create_features(df):
    print("Starting feature engineering...")
    
    feature_df = df.copy()
    
    # Create derived features
    # 1. Spending behavior features
    # Transaction frequency: transaction count / days account has been active
    try:
        if all(col in feature_df.columns for col in ['transaction_amt_count', 'transaction_date_max', 'transaction_date_min']):
            try:
                # Try to convert to date type
                date_max = pd.to_datetime(feature_df['transaction_date_max'])
                date_min = pd.to_datetime(feature_df['transaction_date_min'])
                date_diff = date_max - date_min
                days_active = date_diff.dt.days.fillna(30)  # Fill missing values with 30 days
                # Prevent division by zero
                days_active = days_active.replace({0: 1})
                feature_df['transaction_frequency'] = feature_df['transaction_amt_count'] / days_active
                print("Successfully created transaction_frequency feature")
            except Exception as e:
                print(f"Error creating transaction frequency feature: {e}")
    except Exception as e:
        print(f"Error processing transaction dates: {e}")
    
    # 2. Credit usage features
    # Average credit limit utilization
    if 'ca_current_utilz' in feature_df.columns:
        feature_df['credit_utilization'] = feature_df['ca_current_utilz']
        print("Successfully created credit_utilization feature")
    
    # 3. Risk features
    # Combination of credit score and behavior score
    if all(col in feature_df.columns for col in ['cu_bhv_scr', 'rb_new_bhv_scr']):
        # Standardized combined behavior score
        feature_df['behavior_score_combined'] = (feature_df['cu_bhv_scr'] + feature_df['rb_new_bhv_scr']) / 2
        print("Successfully created behavior_score_combined feature")
    
    if 'cu_crd_bureau_scr' in feature_df.columns:
        # Use external credit score
        feature_df['external_credit_score'] = feature_df['cu_crd_bureau_scr']
        print("Successfully created external_credit_score feature")
    
    # Delinquency risk
    if 'ca_max_dlq_lst_6_mnths' in feature_df.columns:
        feature_df['delinquency_risk'] = feature_df['ca_max_dlq_lst_6_mnths']
        print("Successfully created delinquency_risk feature")
    
    # Fraud risk features
    if 'fraud_case_count' in feature_df.columns:
        # Fraud flag
        feature_df['has_fraud'] = (feature_df['fraud_case_count'] > 0).astype(int)
        print("Successfully created has_fraud feature")
    else:
        # If no fraud data, create default value
        feature_df['has_fraud'] = 0
        print("Created default has_fraud feature")

    # 4. Payment indicators
    # Calculate the ratio of current balance to average previous balance
    if all(col in feature_df.columns for col in ['prev_balance_mean', 'cu_cur_balance']):
        # Prevent division by zero
        denominator = feature_df['prev_balance_mean'].replace({0: 0.01})
        feature_df['balance_ratio'] = feature_df['cu_cur_balance'] / denominator
        print("Successfully created balance_ratio feature")
    
    # 5. Sales growth features
    # If there are multiple months of sales data, create sales growth rate feature
    sales_cols = [col for col in feature_df.columns if 'mo_tot_sales_array' in col]
    if len(sales_cols) >= 2:
        try:
            # Calculate sales growth rate for the last three months
            recent_sales = ['mo_tot_sales_array_1', 'mo_tot_sales_array_2', 'mo_tot_sales_array_3']
            if all(col in feature_df.columns for col in recent_sales):
                # Prevent division by zero
                denominator = feature_df['mo_tot_sales_array_3'].replace({0: 0.01})
                feature_df['sales_growth_rate'] = (feature_df['mo_tot_sales_array_1'] - feature_df['mo_tot_sales_array_3']) / denominator
                print("Successfully created sales_growth_rate feature")
        except Exception as e:
            print(f"Error creating sales growth rate feature: {e}")
    
    # 6. Account activity
    # Account age in months
    if 'ca_mob' in feature_df.columns:
        feature_df['account_age_months'] = feature_df['ca_mob']
        print("Successfully created account_age_months feature")
    
    # Recent activity level
    if 'ca_mnths_since_active' in feature_df.columns:
        feature_df['recent_activity'] = 1 / (feature_df['ca_mnths_since_active'] + 1)  # Prevent division by zero
        print("Successfully created recent_activity feature")
    
    # Print list of created features
    created_features = [col for col in feature_df.columns if col not in df.columns]
    print(f"Feature engineering completed, successfully created features: {created_features}")
    
    return feature_df

# Create derived features
df_with_features = create_features(df)

# Print statistics for newly created features
print("\nBasic statistics for new features:")
for col in df_with_features.columns:
    if col not in df.columns:
        try:
            print(f"{col}: non-null count={df_with_features[col].count()}, mean={df_with_features[col].mean():.4f}")
        except:
            print(f"{col}: non-numeric feature")

# Select features for clustering
print("\nSelecting features for clustering...")
# More flexible feature selection
# First define potential feature categories
potential_features = {
    # Spending behavior indicators
    'Spending Behavior': ['transaction_amt_mean', 'transaction_amt_sum', 'transaction_amt_count', 
                         'transaction_frequency'],
    
    # Credit risk indicators
    'Credit Risk': ['behavior_score_combined', 'external_credit_score', 'delinquency_risk',
                   'cu_nbr_days_dlq', 'cu_cur_nbr_due'],
    
    # Account usage indicators
    'Account Usage': ['credit_utilization', 'ca_avg_utilz_lst_6_mnths', 'ca_avg_utilz_lst_3_mnths',
                     'balance_ratio'],
    
    # Fraud flag
    'Fraud Risk': ['has_fraud'],
    
    # Sales growth
    'Sales Growth': ['sales_growth_rate'],
    
    # Account activity
    'Account Activity': ['account_age_months', 'recent_activity'],
    
    # Other important features
    'Other Features': ['cu_crd_line', 'cu_cur_balance', 'ca_max_dlq_lst_6_mnths', 'prev_balance_mean']
}

# Collect all potential features
all_potential_features = []
for category, features in potential_features.items():
    all_potential_features.extend(features)

# Check which features actually exist in the data
valid_features = [col for col in all_potential_features if col in df_with_features.columns]
print(f"Valid feature list ({len(valid_features)} features): {valid_features}")

# Ensure there are at least some features for clustering
if len(valid_features) < 3:
    print("Warning: Too few valid features, trying to add more basic features...")
    # Add more basic numeric features
    numeric_cols = df_with_features.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Exclude columns not suitable for clustering, such as ID columns
    exclude_patterns = ['_nbr', 'id', 'count']
    additional_features = [col for col in numeric_cols if not any(pattern in col.lower() for pattern in exclude_patterns)]
    valid_features.extend([col for col in additional_features if col not in valid_features])
    valid_features = valid_features[:20]  # Limit maximum number of features
    print(f"Extended feature list: {valid_features}")

# Select valid feature subset
X = df_with_features[valid_features].copy()

# Check and handle infinities and NaN values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())  # Use mean filling instead of 0, which is more reasonable

# View processed data information
print(f"Processed feature matrix shape: {X.shape}")
print("Feature data types:")
print(X.dtypes)

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering algorithm
print("\nApplying K-means clustering algorithm...")
n_clusters = 4  # Four-level classification
# Use correct initialization method to adapt to sklearn 1.0+ version
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to original data
df_with_features['cluster'] = cluster_labels

# Get cluster centers
cluster_centers = kmeans.cluster_centers_

# Analyze features of each cluster
print("\nAnalyzing cluster features...")
cluster_centers_df = pd.DataFrame(cluster_centers, columns=valid_features)
print("Cluster centers:")
print(cluster_centers_df)

# Calculate silhouette coefficient to evaluate clustering effect
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette coefficient: {silhouette_avg}")

# Cluster label interpretation and correspondence determination
print("\nInterpreting clustering results and determining label correspondences...")

# Analyze key feature averages for each cluster
# Dynamically build aggregation dictionary, only include features we have
agg_dict = {}

# Check if key analysis features exist
key_features = [
    # Credit risk indicators
    'behavior_score_combined', 'external_credit_score', 'delinquency_risk', 'cu_nbr_days_dlq',
    
    # Spending indicators
    'transaction_amt_mean', 'transaction_frequency',
    
    # Credit usage indicators
    'credit_utilization',
    
    # Fraud flag
    'has_fraud',
    
    # Account activity
    'recent_activity'
]

# Only include existing columns
for feature in key_features:
    if feature in df_with_features.columns:
        agg_dict[feature] = 'mean'

# Add count column
agg_dict['cluster'] = 'count'

# Check if there are enough indicators
if len(agg_dict) <= 1:  # Not enough if only 'cluster' count
    print("Warning: Not enough key analysis features, will use all valid numeric features...")
    # Use all valid numeric features
    for feature in valid_features:
        if feature in df_with_features.columns and feature != 'cluster':
            agg_dict[feature] = 'mean'

# Calculate cluster statistics
cluster_analysis = df_with_features.groupby('cluster').agg(agg_dict).rename(columns={'cluster': 'count'})
print("Cluster analysis:")
print(cluster_analysis)

# Label mapping logic
# Based on cluster feature distribution, determine which cluster corresponds to which label

# Simplified scoring rule
def determine_cluster_mapping(cluster_analysis):
    # Initialize default mapping
    default_mapping = {i: i for i in range(n_clusters)}
    
    # If not enough features for evaluation, return default mapping
    if cluster_analysis.shape[1] <= 1:
        print("Not enough features for evaluation, using default mapping")
        return default_mapping
    
    # Initialize cluster scores
    cluster_scores = {}
    
    # Determine positive and negative features
    # Positive features: higher is better
    positive_features = [
        'behavior_score_combined', 'external_credit_score', 
        'transaction_amt_mean', 'transaction_frequency',
        'recent_activity', 'credit_utilization'
    ]
    
    # Negative features: lower is better
    negative_features = [
        'delinquency_risk', 'cu_nbr_days_dlq', 'has_fraud'
    ]
    
    try:
        for cluster_id in range(n_clusters):
            total_score = 0
            
            # Calculate score for each feature
            for feature in cluster_analysis.columns:
                if feature == 'count':
                    continue
                
                # Normalize feature value
                if cluster_analysis[feature].max() > 0:
                    normalized_value = cluster_analysis.loc[cluster_id, feature] / cluster_analysis[feature].max()
                    
                    # Add or subtract score based on feature type
                    if feature in positive_features:
                        total_score += normalized_value
                    elif feature in negative_features:
                        total_score -= normalized_value
                    else:
                        # For other features, judge by column name
                        if any(keyword in feature.lower() for keyword in ['score', 'amt', 'frequency', 'activity']):
                            total_score += normalized_value
                        elif any(keyword in feature.lower() for keyword in ['risk', 'dlq', 'fraud', 'due']):
                            total_score -= normalized_value
                
            cluster_scores[cluster_id] = total_score
    except Exception as e:
        print(f"Error calculating cluster scores: {e}")
        return default_mapping
    
    print("Cluster scores:")
    for cluster_id, score in cluster_scores.items():
        print(f"Cluster {cluster_id}: {score:.4f}")
    
    try:
        # Sort clusters by score from high to low
        sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create mapping
        mapping = {}
        # 0: Accounts eligible for credit line increase without risk (highest score)
        # 1: Accounts eligible for credit line increase but with risk (second highest)
        # 2: Accounts where no credit line increase is required (third)
        # 3: Non-performing accounts that pose high risk (lowest score)
        for i, (cluster_id, _) in enumerate(sorted_clusters):
            mapping[cluster_id] = i
        
        return mapping
    except Exception as e:
        print(f"Error creating cluster mapping: {e}")
        return default_mapping

# Determine cluster mapping
cluster_mapping = determine_cluster_mapping(cluster_analysis)
print(f"Cluster mapping: {cluster_mapping}")

# Create label name dictionary
label_names = {
    0: "Eligible for increase without risk",
    1: "Eligible for increase with risk",
    2: "No increase required",
    3: "High risk accounts"
}

# Map clusters to meaningful labels
df_with_features['segment'] = df_with_features['cluster'].map(
    {cluster: label for cluster, label in cluster_mapping.items()}
)
df_with_features['segment_name'] = df_with_features['segment'].map(label_names)

# Save classification results
print("\nSaving classification results...")
output_file = os.path.join(OUTPUT_PATH, 'clustered_accounts.csv')
df_with_features.to_csv(output_file, index=False)
print(f"Clustering results saved to: {output_file}")

try:
    # Create visualization for each account label
    print("\nCreating classification visualization...")
    
    # Use PCA to convert high-dimensional data to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create visualization dataframe
    viz_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    viz_df['segment'] = df_with_features['segment']
    viz_df['segment_name'] = df_with_features['segment_name']
    
    # Save visualization data
    viz_file = os.path.join(OUTPUT_PATH, 'cluster_visualization_data.csv')
    viz_df.to_csv(viz_file, index=False)
    
    # Create scatter plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=viz_df, x='PC1', y='PC2', hue='segment_name', palette='viridis', s=30, alpha=0.6)
    plt.title('Four-Level Account Classification Clustering Results', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Account Category', fontsize=10)
    
    # Save image
    viz_img = os.path.join(OUTPUT_PATH, 'cluster_visualization.png')
    plt.savefig(viz_img, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Visualization image saved to: {viz_img}")
except Exception as e:
    print(f"Error creating visualization: {e}")

try:
    # Feature importance analysis
    print("\nCreating feature importance analysis...")
    
    # Use Random Forest classifier
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
    clf.fit(X_scaled, df_with_features['segment'])
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': valid_features,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature importance Top 10:")
    print(feature_importance.head(10))
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(OUTPUT_PATH, 'feature_importance.csv'), index=False)
except Exception as e:
    print(f"Error calculating feature importance: {e}")

# Count accounts in each label and calculate percentages
segment_counts = df_with_features['segment_name'].value_counts()
segment_percentages = df_with_features['segment_name'].value_counts(normalize=True) * 100

print("\nAccount count statistics by category:")
for segment, count in segment_counts.items():
    percentage = segment_percentages[segment]
    print(f"{segment}: {count} accounts ({percentage:.2f}%)")

print("\nClustering model and analysis completed!")
