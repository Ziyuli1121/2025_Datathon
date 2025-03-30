import pandas as pd
import numpy as np
import os
from datetime import datetime

DATA_PATH = os.path.join('..', 'data', 'data')
OUTPUT_PATH = '.'

os.makedirs(OUTPUT_PATH, exist_ok=True)

def read_csv_with_dtypes(file_path):
    """Read CSV file, infer date columns, and handle parsing errors"""
    try:
        sample_df = pd.read_csv(file_path, nrows=100)
        
        date_cols = []
        for col in sample_df.columns:
            if any(date_word in col.lower() for date_word in ['date', 'dt']):
                date_cols.append(col)
        
        df = pd.read_csv(file_path, parse_dates=date_cols, 
                        low_memory=False, on_bad_lines='skip')
        
        print(f"Successfully read file: {file_path}, rows: {df.shape[0]}, columns: {df.shape[1]}")
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

print("Starting to read data files...")

account_dim = read_csv_with_dtypes(os.path.join(DATA_PATH, 'account_dim_20250325.csv'))
statement_fact = read_csv_with_dtypes(os.path.join(DATA_PATH, 'statement_fact_20250325.csv'))
transaction_fact = read_csv_with_dtypes(os.path.join(DATA_PATH, 'transaction_fact_20250325.csv'))
wrld_stor_tran_fact = read_csv_with_dtypes(os.path.join(DATA_PATH, 'wrld_stor_tran_fact_20250325.csv'))
syf_id = read_csv_with_dtypes(os.path.join(DATA_PATH, 'syf_id_20250325.csv'))
rams_batch_cur = read_csv_with_dtypes(os.path.join(DATA_PATH, 'rams_batch_cur_20250325.csv'))
fraud_claim_case = read_csv_with_dtypes(os.path.join(DATA_PATH, 'fraud_claim_case_20250325.csv'))
fraud_claim_tran = read_csv_with_dtypes(os.path.join(DATA_PATH, 'fraud_claim_tran_20250325.csv'))

print("\nStarting to rename columns...")

rename_cols = {
    'syf_id': {'account_nbr_pty': 'current_account_nbr'},
    'rams_batch_cur': {'cu_account_nbr': 'current_account_nbr'},
    'fraud_claim_case': {'current_account_nbr_pty': 'current_account_nbr'},
    'fraud_claim_tran': {'current_account_nbr_pty': 'current_account_nbr'},
}

for df_name, rename_dict in rename_cols.items():
    if locals()[df_name] is not None:
        locals()[df_name] = locals()[df_name].rename(columns=rename_dict)
        print(f"Already renamed {df_name} table's primary key column to 'current_account_nbr'")

if account_dim is not None:
    print("\nStarting to left join all data tables by account...")
    
    all_accounts = pd.DataFrame({'current_account_nbr': account_dim['current_account_nbr'].unique()})
    print(f"There are {all_accounts.shape[0]} unique accounts in the account_dim table")
    
    account_features = {}
    
    # 1. Add features from account_dim table
    if account_dim is not None:
        account_dim_subset = account_dim[['current_account_nbr', 'client_id', 'open_date', 
                                         'card_activation_date', 'card_activation_flag',
                                         'date_in_collection', 'overlimit_type_flag',
                                         'payment_hist_1_12_mths', 'payment_hist_13_24_mths']]
        account_features['account_dim'] = account_dim_subset
        print("Added account_dim table features")
    
    # 2. Add features from statement_fact table
    if statement_fact is not None:
        # Aggregate statement data
        statement_agg = statement_fact.groupby('current_account_nbr').agg({
            'prev_balance': ['mean', 'max', 'min', 'std', 'count'],
            'billing_cycle_date': ['max', 'min'] # Latest and earliest billing cycle date
        })
        statement_agg.columns = ['_'.join(col).strip() for col in statement_agg.columns.values]
        statement_agg = statement_agg.reset_index()
        
        account_features['statement_fact'] = statement_agg
        print("Added statement_fact table aggregation features")
    
    # 3. Add features from transaction_fact table
    if transaction_fact is not None:
        # Aggregate transaction data
        transaction_agg = transaction_fact.groupby('current_account_nbr').agg({
            'transaction_amt': ['sum', 'mean', 'count', 'max'],
            'transaction_date': ['max', 'min'], # Latest and earliest transaction date
            # Add more transaction features
        })
        transaction_agg.columns = ['_'.join(col).strip() for col in transaction_agg.columns.values]
        transaction_agg = transaction_agg.reset_index()
        
        account_features['transaction_fact'] = transaction_agg
        print("Added transaction_fact table aggregation features")
    
    # 4. Add features from wrld_stor_tran_fact table
    if wrld_stor_tran_fact is not None:
        # Aggregate world transaction data
        wrld_tran_agg = wrld_stor_tran_fact.groupby('current_account_nbr').agg({
            'transaction_amt': ['sum', 'mean', 'count', 'max'],
            'transaction_date': ['max', 'min'], # Latest and earliest transaction date
            # Add more transaction features
        })
        wrld_tran_agg.columns = ['_'.join(col).strip() for col in wrld_tran_agg.columns.values]
        wrld_tran_agg = wrld_tran_agg.reset_index()
        
        account_features['wrld_stor_tran_fact'] = wrld_tran_agg
        print("Added wrld_stor_tran_fact table aggregation features")
    
    # 5. Add features from syf_id table
    if syf_id is not None:
        # Select key columns
        syf_id_subset = syf_id[['current_account_nbr', 'ds_id', 'confidence_level', 
                               'open_date', 'closed_date']]
        account_features['syf_id'] = syf_id_subset
        print("Added syf_id table features")
    
    # 6. Add features from rams_batch_cur table
    if rams_batch_cur is not None:
        # Select key columns
        rams_batch_subset = rams_batch_cur[['current_account_nbr', 'cu_bhv_scr', 'ca_cash_bal_pct_crd_line',
                                           'cu_nbr_days_dlq', 'ca_avg_utilz_lst_6_mnths', 'cu_cash_line_am',
                                           'cu_crd_bureau_scr', 'cu_crd_line', 'cu_cur_balance', 
                                           'cu_cur_nbr_due', 'ca_current_utilz', 'ca_max_dlq_lst_6_mnths',
                                           'rb_new_bhv_scr', 'mo_tot_sales_array_1', 'mo_tot_sales_array_2',
                                           'mo_tot_sales_array_3', 'mo_tot_sales_array_4', 'mo_tot_sales_array_5',
                                           'mo_tot_sales_array_6']]
        account_features['rams_batch_cur'] = rams_batch_subset
        print("Added rams_batch_cur table features")
    
    # 7. Add fraud flag features
    if fraud_claim_case is not None:
        # Create fraud flag
        fraud_flag = fraud_claim_case.groupby('current_account_nbr').size().reset_index()
        fraud_flag.columns = ['current_account_nbr', 'fraud_case_count']
        account_features['fraud_flag'] = fraud_flag
        print("Added fraud_claim_case table fraud flag features")
    
    # Merge all features into the base account table
    merged_data = all_accounts.copy()
    
    for feature_name, feature_df in account_features.items():
        merged_data = pd.merge(merged_data, feature_df, on='current_account_nbr', how='left')
        print(f"Merged {feature_name} table, now {merged_data.shape[1]} columns")
    
    # Handle missing values
    print("\nHandling missing values...")
    
    # Fill numeric features with 0 (adjust as needed)
    num_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns
    merged_data[num_cols] = merged_data[num_cols].fillna(0)
    
    # Fill string features with empty strings
    str_cols = merged_data.select_dtypes(include=['object']).columns
    merged_data[str_cols] = merged_data[str_cols].fillna('')
    
    #########################################################
    # Maybe add more feature engineering if time allows...
    #########################################################
    #########################################################
    
    #
    #
    #



    
    # Save processed data
    output_file = os.path.join(OUTPUT_PATH, 'preprocessed_accounts.csv')
    merged_data.to_csv(output_file, index=False)
    print(f"\nPreprocessing completed! Processed data saved to: {output_file}")
    print(f"Final data shape: {merged_data.shape}")
    
else:
    print("Error: The account_dim table is not available.")

print("\nData preprocessing completed!")