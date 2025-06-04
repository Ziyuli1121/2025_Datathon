import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set input and output paths
CLUSTER_PATH = '../problem2/clustered_accounts.csv'  # Risk classification result file
SPENDING_PATH = '../problem1/q4_2025_spending_predictions.csv'  # Q4 2025 spending prediction result file
OUTPUT_PATH = '.'

###### You can also try other clustering results:
###### KMeans: ../problem2/clustered_accounts.csv
###### MinibatchKMeans: ../problem2/pytorch_models_optimized/mbkmeans_clustered_accounts.csv
###### Birch: ../problem2/pytorch_models_optimized/birch_clustered_accounts.csv
###### Hierarchical: ../problem2/pytorch_models_optimized/hierarchical_clustered_accounts.csv
###### DBSCAN: ../problem2/pytorch_models_optimized/dbscan_clustered_accounts.csv

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Configure logging to file
log_file_name = f"credit_line_recommendation_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file_path = os.path.join(OUTPUT_PATH, log_file_name)

# Create a class to capture stdout and write to both stdout and log file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure log is written immediately
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Store original stdout for restoration later
original_stdout = sys.stdout

# Redirect stdout to our logger
sys.stdout = Logger(log_file_path)

print(f"Starting credit line recommendation process - Log file: {log_file_path}")
print(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 80)

print("Loading classification data and spending prediction data...")

# Load classification data
try:
    df_segments = pd.read_csv(CLUSTER_PATH)
    print(f"Successfully loaded classification data, shape: {df_segments.shape}")
    # Display column names for debugging
    print(f"Classification data columns: {df_segments.columns.tolist()}")
except Exception as e:
    print(f"Failed to load classification data: {e}")
    # Try to find other classification result files
    alt_cluster_files = [
        'clustered_accounts.csv',
        'pytorch_models_optimized/mbkmeans_clustered_accounts.csv',
        'pytorch_models_optimized/birch_clustered_accounts.csv', 
        'pytorch_models_optimized/hierarchical_clustered_accounts.csv',
        'pytorch_models_optimized/dbscan_clustered_accounts.csv'
    ]
    
    for file in alt_cluster_files:
        if os.path.exists(file):
            print(f"Found alternative classification file: {file}")
            df_segments = pd.read_csv(file)
            print(f"Classification data columns: {df_segments.columns.tolist()}")
            break
    else:
        print("Cannot find classification data file, exiting program")
        exit(1)

# Load spending prediction data
try:
    df_spending = pd.read_csv(SPENDING_PATH)
    print(f"Successfully loaded spending prediction data, shape: {df_spending.shape}")
    print(f"Spending prediction columns: {df_spending.columns.tolist()}")
    
    # Data type conversion: ensure prediction columns are numeric with careful handling
    for col in df_spending.columns:
        if col != 'current_account_nbr' and 'account' not in col.lower():
            try:
                # Check for non-numeric values before conversion
                non_numeric = pd.to_numeric(df_spending[col], errors='coerce').isna().sum()
                if non_numeric > 0:
                    print(f"Warning: Column {col} has {non_numeric} non-numeric values")
                
                # Convert with careful handling
                df_spending[col] = pd.to_numeric(df_spending[col], errors='coerce')
                # Fill NaN values with median to avoid data loss
                if df_spending[col].isna().sum() > 0:
                    df_spending[col] = df_spending[col].fillna(df_spending[col].median())
                print(f"Converted column {col} to numeric")
            except Exception as e:
                print(f"Warning: Unable to convert column {col} to numeric: {e}")
    
except Exception as e:
    print(f"Failed to load spending prediction data: {e}")
    # Try to find other prediction result files
    alt_forecast_files = [
        '../problem1/q4_2025_spending_predictions.csv',
        '../problem1/q4_2025_account_totals.csv',
        '../problem1/q2_q3_q4_2025_predictions.csv'
    ]
    
    for file in alt_forecast_files:
        if os.path.exists(file):
            print(f"Found alternative prediction file: {file}")
            df_spending = pd.read_csv(file)
            print(f"Spending prediction columns: {df_spending.columns.tolist()}")
            
            # Data type conversion with careful handling
            for col in df_spending.columns:
                if col != 'current_account_nbr' and 'account' not in col.lower():
                    try:
                        df_spending[col] = pd.to_numeric(df_spending[col], errors='coerce')
                        # Fill NaN values with median
                        if df_spending[col].isna().sum() > 0:
                            df_spending[col] = df_spending[col].fillna(df_spending[col].median())
                        print(f"Converted column {col} to numeric")
                    except Exception as e:
                        print(f"Warning: Unable to convert column {col} to numeric: {e}")
            
            break
    else:
        print("Warning: Cannot find spending prediction data file, will generate recommendations based on classification results only")
        df_spending = None

# Merge datasets
print("Merging datasets...")
if df_spending is not None:
    # Determine the primary key column names based on documented column names
    # According to data_mapping_document.markdown, the primary keys are:
    # - current_account_nbr (most tables)
    # - cu_account_nbr (rams_batch_cur table)
    # - current_account_nbr_pty (fraud tables)
    
    account_cols_segments = [col for col in df_segments.columns 
                             if col in ['current_account_nbr', 'cu_account_nbr', 'current_account_nbr_pty']]
    
    account_cols_spending = [col for col in df_spending.columns 
                              if col in ['current_account_nbr', 'cu_account_nbr', 'current_account_nbr_pty']]
    
    if account_cols_segments and account_cols_spending:
        account_col_segments = account_cols_segments[0]
        account_col_spending = account_cols_spending[0]
        
        print(f"Merging datasets using primary keys: {account_col_segments}(classification) and {account_col_spending}(prediction)")
        df_segments = df_segments.rename(columns={account_col_segments: 'account_key'})
        df_spending = df_spending.rename(columns={account_col_spending: 'account_key'})
        df_combined = pd.merge(df_segments, df_spending, on='account_key', how='left')
        # Restore original column names
        df_combined = df_combined.rename(columns={'account_key': account_col_segments})
        print(f"Merged dataset shape: {df_combined.shape}")
    else:
        print(f"Warning: Unable to find matching account columns. Classification columns: {account_cols_segments}, Prediction columns: {account_cols_spending}")
        print("Will use classification data only")
        df_combined = df_segments.copy()
else:
    df_combined = df_segments.copy()

# Credit line increase recommendation function
def recommend_credit_line_increase(df):
    """
    Generate credit line increase recommendations for each account
    
    Strategy:
    1. For "can increase credit line without risk" category - provide the highest increase
    2. For "can increase credit line with risk" category - provide a smaller increase
    3. For "no need to increase credit line" category - do not provide an increase
    4. For "high risk accounts" category - do not provide an increase, may consider reducing the limit
    """
    print("Generating credit line increase recommendations...")
    print(f"Input dataframe columns: {df.columns.tolist()}")
    
    # Prepare result dataframe
    df_result = df.copy()
    
    # Add credit line increase recommendation columns
    df_result['credit_line_increase_amount'] = 0.0
    df_result['credit_line_increase_percentage'] = 0.0
    df_result['recommendation_reason'] = ''
    
    # Get current credit line column (use documented column name from data mapping)
    if 'cu_crd_line' in df.columns:
        print("Found credit line column: cu_crd_line")
        df_result['current_credit_line'] = df['cu_crd_line']
    else:
        # Fallback to other potential credit line columns if the primary one isn't found
        credit_line_cols = [col for col in df.columns if 'crd_line' in col.lower() or 'credit_limit' in col.lower()]
        if credit_line_cols:
            credit_line_col = credit_line_cols[0]
            print(f"Using alternative credit line column: {credit_line_col}")
            df_result['current_credit_line'] = df[credit_line_col]
        else:
            print("Warning: Cannot find credit line column, using default value 10000")
            df_result['current_credit_line'] = 10000
    
    # Handle potential NaN values in current_credit_line
    df_result['current_credit_line'] = df_result['current_credit_line'].fillna(10000)
    
    # Get credit score columns (use documented column names)
    score_cols = []
    for col_name in ['cu_bhv_scr', 'cu_crd_bureau_scr', 'rb_new_bhv_scr']:
        if col_name in df.columns:
            score_cols.append(col_name)
            print(f"Found score column: {col_name}")
    
    has_score = len(score_cols) > 0
    
    # Get credit utilization columns (use documented column names)
    util_cols = []
    for col_name in ['ca_current_utilz', 'ca_avg_utilz_lst_3_mnths', 'ca_avg_utilz_lst_6_mnths']:
        if col_name in df.columns:
            util_cols.append(col_name)
            print(f"Found utilization column: {col_name}")
    
    has_util = len(util_cols) > 0
    
    # Get spending prediction column
    forecast_cols = [col for col in df.columns if 'forecast' in col.lower() or 'prediction' in col.lower() or 'q4_' in col.lower()]
    has_forecast = len(forecast_cols) > 0
    
    if has_forecast:
        print(f"Found spending prediction column: {forecast_cols[0]}")
        # Ensure prediction column is numeric with proper error handling
        try:
            df_result[forecast_cols[0]] = pd.to_numeric(df_result[forecast_cols[0]], errors='coerce')
            # Fill NA values with median to avoid data loss
            median_val = df_result[forecast_cols[0]].median()
            df_result[forecast_cols[0]] = df_result[forecast_cols[0]].fillna(median_val)
            print(f"Converted prediction column {forecast_cols[0]} to numeric")
        except Exception as e:
            print(f"Error converting prediction column {forecast_cols[0]}: {e}")
            has_forecast = False
    
    # Check segment column (segment or cluster)
    if 'segment' in df.columns:
        segment_col = 'segment'
    elif 'cluster' in df.columns:
        segment_col = 'cluster'
    else:
        # Find any column that might contain segment information
        potential_segment_cols = [col for col in df.columns if 'segment' in col.lower() or 'cluster' in col.lower() or 'group' in col.lower()]
        segment_col = potential_segment_cols[0] if potential_segment_cols else None
        
        if not segment_col:
            print("Warning: Cannot find segment column, creating default segment")
            # Create a default segment based on credit utilization if available
            if has_util:
                df_result['segment'] = df_result[util_cols[0]].apply(
                    lambda x: 0 if x < 30 else (1 if x < 60 else (2 if x < 80 else 3)))
                segment_col = 'segment'
            else:
                # Random assignment as last resort
                df_result['segment'] = np.random.randint(0, 4, size=len(df_result))
                segment_col = 'segment'
    
    segment_name_col = 'segment_name' if 'segment_name' in df.columns else None
    print(f"Using segment column: {segment_col}")
    
    # Define rules for each category
    for index, row in df_result.iterrows():
        current_credit_line = row['current_credit_line']
        
        # Get account category
        if segment_name_col and segment_name_col in df.columns:
            segment = row[segment_name_col]
        else:
            segment = row[segment_col]  # Use numeric category
        
        # Default increase percentage and reason
        increase_percentage = 0.0
        reason = "Default strategy"
        
        # Apply different rules based on account category
        if isinstance(segment, str):
            # Use string category name
            if any(term in segment.lower() for term in ["without risk", "low risk", "safe", "good"]):
                # For no-risk customers, provide a higher increase
                base_increase = 0.25  # Base increase 25%
                
                # Increase factors
                if has_score:
                    # Use the first available score
                    score_col = score_cols[0]
                    if row[score_col] > df[score_col].median():
                        base_increase += 0.10  # High score adds 10%
                        reason = f"No-risk customer, high {score_col}"
                    else:
                        reason = "No-risk customer, standard increase"
                else:
                    reason = "No-risk customer, standard increase"
                
                # Adjust based on spending prediction if available
                if has_forecast and not pd.isna(row[forecast_cols[0]]):
                    try:
                        forecast_value = float(row[forecast_cols[0]])
                        if forecast_value > current_credit_line * 0.7:
                            base_increase += 0.10  # High spending prediction adds 10%
                            reason += ", high spending prediction"
                    except (ValueError, TypeError) as e:
                        # Ignore conversion errors
                        pass
                
                increase_percentage = base_increase
                
            elif any(term in segment.lower() for term in ["with risk", "moderate risk", "medium"]):
                # For at-risk customers, provide a moderate increase
                base_increase = 0.10  # Base increase 10%
                
                # Adjust based on score
                if has_score:
                    # Use the first available score
                    score_col = score_cols[0]
                    if row[score_col] > df[score_col].median():
                        base_increase += 0.05  # High score adds 5%
                        reason = f"At-risk customer, relatively good {score_col}"
                    else:
                        reason = "At-risk customer, cautious increase"
                else:
                    reason = "At-risk customer, cautious increase"
                
                # Adjust based on utilization
                if has_util:
                    # Use the first available utilization metric
                    util_col = util_cols[0]
                    if row[util_col] < 50:  # Assuming utilization is in percentage (0-100)
                        base_increase += 0.05  # Low utilization adds 5%
                        reason += f", low {util_col}"
                
                increase_percentage = base_increase
                
            elif any(term in segment.lower() for term in ["no need", "no increase", "neutral"]):
                # No need to increase credit line
                increase_percentage = 0.0
                reason = "No need to increase credit line"
                
            elif any(term in segment.lower() for term in ["high risk", "bad", "poor"]):
                # High-risk accounts do not increase credit line
                increase_percentage = 0.0
                reason = "High-risk account, not recommended to increase credit line"
            else:
                # For any other string category not matching our patterns
                print(f"Unrecognized segment value: {segment}, treating as neutral")
                increase_percentage = 0.0
                reason = "Unrecognized segment, conservative approach"
        else:
            # Use numeric category (assuming 0=best, 3=worst)
            try:
                segment_num = int(segment)
                if segment_num == 0:  # No-risk customer
                    base_increase = 0.25
                    if has_score:
                        score_col = score_cols[0]
                        if row[score_col] > df[score_col].median():
                            base_increase += 0.10
                            reason = f"No-risk customer, high {score_col}"
                        else:
                            reason = "No-risk customer, standard increase"
                    else:
                        reason = "No-risk customer, standard increase"
                        
                    if has_forecast and not pd.isna(row[forecast_cols[0]]):
                        try:
                            forecast_value = float(row[forecast_cols[0]])
                            if forecast_value > current_credit_line * 0.7:
                                base_increase += 0.10
                                reason += ", high spending prediction"
                        except (ValueError, TypeError) as e:
                            # Ignore conversion errors
                            pass
                            
                    increase_percentage = base_increase
                    
                elif segment_num == 1:  # At-risk customer
                    base_increase = 0.10
                    if has_score:
                        score_col = score_cols[0]
                        if row[score_col] > df[score_col].median():
                            base_increase += 0.05
                            reason = f"At-risk customer, relatively good {score_col}"
                        else:
                            reason = "At-risk customer, cautious increase"
                    else:
                        reason = "At-risk customer, cautious increase"
                        
                    if has_util:
                        util_col = util_cols[0]
                        if row[util_col] < 50:  # Assuming utilization is in percentage (0-100)
                            base_increase += 0.05
                            reason += f", low {util_col}"
                        
                    increase_percentage = base_increase
                    
                elif segment_num == 2:  # No need to increase credit line
                    increase_percentage = 0.0
                    reason = "No need to increase credit line"
                    
                elif segment_num == 3:  # High-risk account
                    increase_percentage = 0.0
                    reason = "High-risk account, not recommended to increase credit line"
                else:
                    # For any other numeric category not in 0-3
                    print(f"Unrecognized segment value: {segment}, treating as neutral")
                    increase_percentage = 0.0
                    reason = "Unrecognized segment value, conservative approach"
            except (ValueError, TypeError):
                # Failed to convert to int, treat as neutral
                print(f"Non-numeric segment value that is not a string: {segment}, treating as neutral")
                increase_percentage = 0.0
                reason = "Invalid segment value, conservative approach"
        
        # Calculate recommended increase amount
        increase_amount = current_credit_line * increase_percentage
        
        # Set minimum and maximum increase amounts
        if increase_amount > 0:
            increase_amount = max(increase_amount, 1000)  # Minimum increase 1000
            increase_amount = min(increase_amount, 10000)  # Maximum increase 10000
            
            # Round to the nearest 100
            increase_amount = round(increase_amount / 100) * 100
        
        # Update result
        df_result.at[index, 'credit_line_increase_amount'] = increase_amount
        df_result.at[index, 'credit_line_increase_percentage'] = increase_percentage
        df_result.at[index, 'recommendation_reason'] = reason
    
    return df_result

# Use machine learning model to predict optimal credit line increase
def predict_optimal_increase_with_ml(df):
    """
    Use machine learning model to predict optimal credit line increase
    """
    print("Using machine learning model to predict optimal credit line increase...")
    
    # Define features based on documented column names
    feature_cols = []
    
    # Credit line features
    if 'cu_crd_line' in df.columns:
        feature_cols.append('cu_crd_line')
    
    # Utilization features
    for col in ['ca_current_utilz', 'ca_avg_utilz_lst_3_mnths', 'ca_avg_utilz_lst_6_mnths']:
        if col in df.columns:
            feature_cols.append(col)
    
    # Score features
    for col in ['cu_bhv_scr', 'cu_crd_bureau_scr', 'rb_new_bhv_scr']:
        if col in df.columns:
            feature_cols.append(col)
    
    # Spending prediction features
    forecast_cols = [col for col in df.columns if 'forecast' in col.lower() or 'prediction' in col.lower() or 'q4_' in col.lower()]
    if forecast_cols:
        feature_cols.extend(forecast_cols)
    
    # Segment feature
    if 'segment' in df.columns:
        feature_cols.append('segment')
    elif 'cluster' in df.columns:
        feature_cols.append('cluster')
    
    # Additional potentially useful features if available
    for col in df.columns:
        if col in ['ca_mob', 'ca_max_dlq_lst_6_mnths', 'ca_mnths_since_cl_chng', 'cu_cur_balance']:
            feature_cols.append(col)
    
    # Ensure unique feature list
    feature_cols = list(set(feature_cols))
    print(f"Selected features: {feature_cols}")
    
    # Check if there are enough features
    if len(feature_cols) < 3:
        print("Warning: Not enough features, cannot train machine learning model, will use rule-based method only")
        return None
    
    # Prepare data
    try:
        X = df[feature_cols].copy()
    except KeyError as e:
        print(f"Error selecting features: {e}")
        print("Will use rule-based method only")
        return None
    
    # Handle missing columns
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        print(f"Warning: Some feature columns are missing from the dataframe: {missing_cols}")
        # Remove missing columns from feature list
        feature_cols = [col for col in feature_cols if col in df.columns]
        X = df[feature_cols].copy()
    
    # Ensure all feature columns are numeric
    segment_col = 'segment' if 'segment' in feature_cols else ('cluster' if 'cluster' in feature_cols else None)
    
    for col in X.columns:
        if col != segment_col:  # All columns except the category column should be numeric
            try:
                non_numeric = pd.to_numeric(X[col], errors='coerce').isna().sum()
                if non_numeric > 0:
                    print(f"Column {col} has {non_numeric} non-numeric values")
                
                X[col] = pd.to_numeric(X[col], errors='coerce')
                
                # Handle NaN values - use median imputation
                if X[col].isna().sum() > 0:
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)
                    print(f"Filled {X[col].isna().sum()} NaN values in {col} with median {median_val}")
            except Exception as e:
                print(f"Warning: Error processing column {col}: {e}")
                # Use mean or more robust imputation
                X[col] = X[col].fillna(X[col].mean())
    
    # Handle categorical variables
    if segment_col:
        # Check if segment_col is already numeric
        if X[segment_col].dtype.kind in 'ifc':
            # Already numeric, no need to encode
            pass
        else:
            # Convert categorical to numeric first
            X[segment_col] = pd.Categorical(X[segment_col]).codes
            
        # One-hot encode
        X = pd.get_dummies(X, columns=[segment_col], drop_first=True)
    
    # Create target variable (use the rule-based method result as the "label")
    rule_based_result = recommend_credit_line_increase(df)
    y = rule_based_result['credit_line_increase_amount']
    
    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train random forest regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model evaluation - MSE: {mse:.2f}, R^2: {r2:.4f}")
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 feature importance:")
    print(feature_importance.head(10))
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(OUTPUT_PATH, 'credit_line_increase_feature_importance.csv'), index=False)
    
    # Predict for all data
    df_result = df.copy()
    
    # Prepare data for prediction
    X_all = df[feature_cols].copy()
    
    # Ensure all feature columns are numeric
    for col in X_all.columns:
        if col != segment_col:
            try:
                X_all[col] = pd.to_numeric(X_all[col], errors='coerce')
                # Fill NaN values with median
                if X_all[col].isna().sum() > 0:
                    X_all[col] = X_all[col].fillna(X_all[col].median())
            except Exception as e:
                print(f"Warning: Error processing prediction column {col}: {e}")
                X_all[col] = X_all[col].fillna(X_all[col].mean())
    
    # Handle categorical variables for prediction
    if segment_col:
        if X_all[segment_col].dtype.kind in 'ifc':
            # Already numeric, no need to encode
            pass
        else:
            # Convert categorical to numeric first
            X_all[segment_col] = pd.Categorical(X_all[segment_col]).codes
        
        # One-hot encode
        X_all = pd.get_dummies(X_all, columns=[segment_col], drop_first=True)
    
    # Ensure test and training sets have the same columns
    missing_cols = set(X_train.columns) - set(X_all.columns)
    for col in missing_cols:
        X_all[col] = 0
    
    # Ensure columns are in the same order
    X_all = X_all[X_train.columns]
    
    # Make predictions
    df_result['ml_credit_line_increase'] = model.predict(X_all)
    
    # Adjust prediction results
    df_result['ml_credit_line_increase'] = df_result['ml_credit_line_increase'].apply(
        lambda x: max(0, min(round(x / 100) * 100, 10000))  # Range 0-10000, round to nearest 100
    )
    
    return df_result

# Combine rule-based and machine learning results
def combine_recommendation_results(rule_based_df, ml_df=None):
    """Combine rule-based and machine learning recommendation results"""
    print("Combining recommendation results...")
    
    result_df = rule_based_df.copy()
    
    if ml_df is not None and 'ml_credit_line_increase' in ml_df.columns:
        result_df['ml_credit_line_increase'] = ml_df['ml_credit_line_increase']
        
        # Calculate final recommendation (strategy can be adjusted as needed)
        result_df['final_recommendation'] = result_df.apply(
            lambda row: row['ml_credit_line_increase'] if row['ml_credit_line_increase'] > 0 else row['credit_line_increase_amount'],
            axis=1
        )
    else:
        result_df['final_recommendation'] = result_df['credit_line_increase_amount']
    
    return result_df

# Create visualizations
def create_visualizations(df):
    """Create visualizations for credit line increase recommendations"""
    print("Creating visualizations...")
    
    # Ensure we have the segment column
    if 'segment' in df.columns:
        segment_col = 'segment'
    elif 'cluster' in df.columns:
        segment_col = 'cluster'
    elif 'segment_name' in df.columns:
        segment_col = 'segment_name'
    else:
        # Try to find any column that might contain segment information
        potential_cols = [col for col in df.columns if 'segment' in col.lower() or 'cluster' in col.lower() or 'group' in col.lower()]
        segment_col = potential_cols[0] if potential_cols else None
        
    if not segment_col:
        print("Warning: No segment column found for visualization, creating a default one")
        # Create a default segment based on the final recommendation
        df['visualization_segment'] = pd.qcut(df['final_recommendation'], 4, labels=["Low", "Medium-Low", "Medium-High", "High"])
        segment_col = 'visualization_segment'
    
    print(f"Using segment column for visualization: {segment_col}")
    
    # 1. Average recommended increase amount by category
    plt.figure(figsize=(12, 6))
    # Convert segment to string to ensure it's treated as categorical
    df[segment_col] = df[segment_col].astype(str)
    avg_by_segment = df.groupby(segment_col)['final_recommendation'].mean().sort_values(ascending=False)
    
    # Handle potential NaN or infinite values
    avg_by_segment = avg_by_segment.fillna(0)
    
    sns.barplot(x=avg_by_segment.index, y=avg_by_segment.values)
    plt.title('Average Credit Line Increase Recommendation by Account Category')
    plt.xlabel('Account Category')
    plt.ylabel('Average Recommended Increase Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'avg_increase_by_segment.png'), dpi=200)
    plt.close()
    
    # 2. Distribution of recommended increase amounts
    plt.figure(figsize=(10, 6))
    # Remove any potential NaN values
    final_recommendations = df['final_recommendation'].fillna(0)
    sns.histplot(final_recommendations, bins=20, kde=True)
    plt.title('Distribution of Credit Line Increase Recommendations')
    plt.xlabel('Recommended Increase Amount')
    plt.ylabel('Number of Accounts')
    plt.savefig(os.path.join(OUTPUT_PATH, 'increase_amount_distribution.png'), dpi=200)
    plt.close()
    
    # 3. Scatter plot of current credit line vs recommended increase amount
    if 'current_credit_line' in df.columns:
        plt.figure(figsize=(10, 6))
        # Remove any rows with NaN values for plotting
        plot_df = df[['current_credit_line', 'final_recommendation', segment_col]].copy()
        plot_df = plot_df.dropna()
        
        # Ensure all values are valid for plotting
        plot_df['current_credit_line'] = plot_df['current_credit_line'].replace([np.inf, -np.inf], np.nan).fillna(0)
        plot_df['final_recommendation'] = plot_df['final_recommendation'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        sns.scatterplot(data=plot_df, x='current_credit_line', y='final_recommendation', hue=segment_col, alpha=0.6)
        plt.title('Current Credit Line vs Recommended Increase Amount')
        plt.xlabel('Current Credit Line')
        plt.ylabel('Recommended Increase Amount')
        plt.savefig(os.path.join(OUTPUT_PATH, 'current_vs_increase.png'), dpi=200)
        plt.close()
    
    # 4. Distribution of increase percentages with safe division
    plt.figure(figsize=(10, 6))
    
    # Calculate percentage safely (avoid division by zero)
    df['increase_percentage'] = df.apply(
        lambda row: (row['final_recommendation'] / row['current_credit_line'] * 100) 
                    if row['current_credit_line'] > 0 else 0,
        axis=1
    )
    
    # Handle potential infinity or NaN values
    df['increase_percentage'] = df['increase_percentage'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Clip extreme values for better visualization
    df['increase_percentage'] = df['increase_percentage'].clip(0, 100)
    
    sns.boxplot(x=segment_col, y='increase_percentage', data=df)
    plt.title('Credit Line Increase Percentage by Account Category')
    plt.xlabel('Account Category')
    plt.ylabel('Increase Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'increase_percentage_by_segment.png'), dpi=200)
    plt.close()

def create_data_exploration_visualizations(df_combined, df_segments):
    """Create data exploration visualizations"""
    print("Creating data exploration visualizations...")
    
    # View customer segment distribution
    plt.figure(figsize=(10, 6))
    segment_col = 'cluster' if 'cluster' in df_segments.columns else 'segment'
    segment_counts = df_segments[segment_col].value_counts()
    plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%')
    plt.title('Customer Segment Distribution')
    plt.savefig(os.path.join(OUTPUT_PATH, 'customer_segment_distribution.png'), dpi=200)
    plt.close()

    # Credit line distribution
    if 'cu_crd_line' in df_combined.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_combined['cu_crd_line'].fillna(0), bins=30, kde=True)
        plt.title('Current Credit Line Distribution')
        plt.xlabel('Credit Line')
        plt.ylabel('Number of Customers')
        plt.savefig(os.path.join(OUTPUT_PATH, 'credit_line_distribution.png'), dpi=200)
        plt.close()

    # Credit score distribution
    score_cols = [col for col in df_combined.columns if col in ['cu_bhv_scr', 'cu_crd_bureau_scr', 'rb_new_bhv_scr']]
    if score_cols:
        score_col = score_cols[0]
        plt.figure(figsize=(10, 6))
        sns.histplot(df_combined[score_col].fillna(0), bins=30, kde=True)
        plt.title(f'{score_col} Distribution')
        plt.xlabel('Credit Score')
        plt.ylabel('Number of Customers')
        plt.savefig(os.path.join(OUTPUT_PATH, 'credit_score_distribution.png'), dpi=200)
        plt.close()

def create_feature_correlation_visualizations(df_combined):
    """Create feature correlation visualizations"""
    print("Creating feature correlation visualizations...")
    
    # Calculate correlation of numeric features
    numeric_cols = df_combined.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:  # Ensure at least two numeric columns to calculate correlation
        corr_matrix = df_combined[numeric_cols].corr()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'feature_correlation_heatmap.png'), dpi=200)
        plt.close()

def create_model_evaluation_visualizations(X, y_test, y_pred, model):
    """Create model evaluation visualizations"""
    print("Creating model evaluation visualizations...")
    
    # Model prediction vs actual values
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest Model Prediction vs Actual Values')
    plt.savefig(os.path.join(OUTPUT_PATH, 'model_prediction_vs_actual.png'), dpi=200)
    plt.close()

    # Residual plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    sns.histplot(residuals, bins=30, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Prediction Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(OUTPUT_PATH, 'prediction_residuals.png'), dpi=200)
    plt.close()

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 10))
        sorted_idx = model.feature_importances_.argsort()
        plt.barh(X.columns[sorted_idx], model.feature_importances_[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Model Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'feature_importance_detailed.png'), dpi=200)
        plt.close()

def create_segment_comparison_visualizations(final_result):
    """Create segment comparison visualizations"""
    print("Creating segment comparison visualizations...")
    
    # Determine segment column
    segment_col = None
    for col in ['segment', 'cluster', 'segment_name', 'visualization_segment']:
        if col in final_result.columns:
            segment_col = col
            break
    
    if segment_col:
        # Comparison of current credit line and recommended increase amount by segment
        plt.figure(figsize=(14, 8))
        comparison_data = final_result.groupby(segment_col).agg({
            'current_credit_line': 'mean',
            'final_recommendation': 'mean'
        }).reset_index()

        x = np.arange(len(comparison_data))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - width/2, comparison_data['current_credit_line'], width, label='Current Credit Line')
        ax.bar(x + width/2, comparison_data['final_recommendation'], width, label='Recommended Increase Amount')

        ax.set_xticks(x)
        ax.set_xticklabels(comparison_data[segment_col])
        ax.set_xlabel('Customer Segment')
        ax.set_ylabel('Amount')
        ax.set_title('Comparison of Current Credit Line and Recommended Increase Amount by Segment')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'segment_comparison.png'), dpi=200)
        plt.close()

# Main function
def main():
    print("Starting to generate credit line increase recommendations...")
    
    # 1. Use rule-based method
    rule_based_result = recommend_credit_line_increase(df_combined)
    
    # 2. Try to use machine learning model
    try:
        ml_result = predict_optimal_increase_with_ml(df_combined)
    except Exception as e:
        print(f"Failed to train machine learning model: {e}")
        import traceback
        traceback.print_exc()
        ml_result = None
    
    # 3. Combine results
    final_result = combine_recommendation_results(rule_based_result, ml_result)
    
    # 4. Create basic visualizations
    try:
        create_visualizations(final_result)
    except Exception as e:
        print(f"Failed to create visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Create additional visualizations
    try:
        print("\nCreating additional data visualizations...")
        create_data_exploration_visualizations(df_combined, df_segments)
        create_feature_correlation_visualizations(df_combined)
        
        # Only create model evaluation visualizations if ML model was successfully trained
        if ml_result is not None and 'X' in locals() and 'y_test' in locals() and 'y_pred' in locals() and 'model' in locals():
            create_model_evaluation_visualizations(X, y_test, y_pred, model)
        else:
            print("Machine learning model evaluation visualizations skipped - model or necessary variables not available")
        
        create_segment_comparison_visualizations(final_result)
        print("Additional visualizations created!")
    except Exception as e:
        print(f"Error creating additional visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Save results
    output_file = os.path.join(OUTPUT_PATH, 'credit_line_recommendations.csv')
    final_result.to_csv(output_file, index=False)
    print(f"Credit line increase recommendations saved to: {output_file}")
    
    # 7. Output summary statistics
    # Determine segment column
    if 'segment' in final_result.columns:
        segment_col = 'segment'
    elif 'cluster' in final_result.columns:
        segment_col = 'cluster'
    elif 'segment_name' in final_result.columns:
        segment_col = 'segment_name'
    else:
        potential_cols = [col for col in final_result.columns if 'segment' in col.lower() or 'cluster' in col.lower() or 'group' in col.lower()]
        segment_col = potential_cols[0] if potential_cols else 'visualization_segment'
    
    print(f"\nSummary of credit line increase recommendations by account category (using {segment_col}):")
    
    # Convert segment to string to ensure it's treated as categorical
    final_result[segment_col] = final_result[segment_col].astype(str)
    
    # Handle potential NaN or infinite values in increase_percentage
    if 'increase_percentage' not in final_result.columns:
        final_result['increase_percentage'] = final_result.apply(
            lambda row: (row['final_recommendation'] / row['current_credit_line'] * 100) 
                        if row['current_credit_line'] > 0 else 0,
            axis=1
        )
    
    final_result['increase_percentage'] = final_result['increase_percentage'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Group by segment and calculate statistics
    segment_stats = final_result.groupby(segment_col).agg({
        'final_recommendation': ['count', 'mean', 'min', 'max'],
        'increase_percentage': ['mean', 'min', 'max']
    })
    
    print(segment_stats)
    
    # Overall recommendation statistics
    total_accounts = len(final_result)
    accounts_with_increase = (final_result['final_recommendation'] > 0).sum()
    
    # Safely calculate average increase (avoid division by zero)
    if accounts_with_increase > 0:
        average_increase = final_result[final_result['final_recommendation'] > 0]['final_recommendation'].mean()
    else:
        average_increase = 0
    
    print(f"\nTotal number of accounts: {total_accounts}")
    print(f"Number of accounts recommended for credit line increase: {accounts_with_increase} ({accounts_with_increase/total_accounts*100:.2f}%)")
    print(f"Average recommended increase amount: {average_increase:.2f}")
    
    print("\nCredit line increase recommendation generation completed!")
    
    # Record end time and log file location
    print("-" * 80)
    print(f"Process completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All outputs have been logged to: {log_file_path}")
    
    # Close log file and restore stdout
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    sys.stdout = original_stdout
    print(f"Log file saved to: {log_file_path}")

if __name__ == "__main__":
    main()
