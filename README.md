# Team: Canes

## Presentation Video Link
Our presentation video link: [Team: Canes | UIUC Datathon 2025](https://youtu.be/oTbIIaauKmk)

## File Structure

```bash
Datathon/
├── data/
│   ├── data/   # 8 csv files
│   ├── Datathon Mapping document_V4.xlsx
│   └── Datathon_workshop_guide_0325.docx
├── problem1/
│   ├── problem1.ipynb
│   └── [outputs]
├── problem2/
│   ├── data_preprocessing.py
│   ├── clustering_model.py
│   ├── pytorch_clustering_models_optimized.py
│   └── [outputs]
├── problem3/
│   ├── credit_line_recommendation.py
│   └── [outputs]
├── README.md   # this file
├── Canes.pptx  # presentation slides
├── requirements.txt # environment requirements
└── problem_statement.markdown # summary of the problems

```

## Environment Setup

### 1. Install Anaconda or Miniconda

Download and install the version appropriate for your system from the [Anaconda website](https://www.anaconda.com/products/individual) or [Miniconda website](https://docs.conda.io/en/latest/miniconda.html).

### 2. Create a New Conda Environment

```bash
conda create -n datathon python=3.9
conda activate datathon
```

### 3. Install Dependencies

```bash
conda install pandas=2.2.3 numpy=1.26.4 matplotlib=3.9.2 scikit-learn=1.6.1 seaborn=0.13.2 joblib=1.4.2
pip install statsmodels
pip install xgboost
conda install pytorch=2.5.1 -c pytorch
```

#### PyTorch GPU Support (NVIDIA Graphics Card)

Note: we tested our code on CPU (on macOS-arm64), NVIDIA RTX 4060 (on Windows 11 x64) and NVIDIA A100 (on Linux x86_64). Since we just use very simple NN layers for our autoencoder, there is no significant difference in calculation time. 

```bash
# For systems with CUDA support (NVIDIA GPU)
conda install pytorch=2.5.1 cudatoolkit=11.8 -c pytorch
```


## Reproduction Guide

If there is any problem in your reproduction process, feel free to contact me at [ziyul6@illinois.edu]


```bash
git clone https://github.com/Ziyuli1121/2025_Datathon
```

Firstly, make sure the directories are put in correct location.

```bash
Datathon/
├── data/
│   └── data   # 8 csv files
│  
├── problem1/
│   └── problem1.ipynb
│   
├── problem2/
│   ├── data_preprocessing.py
│   ├── clustering_model.py
│   └── ...
│
├── problem3/
│   ├── credit_line_recommendation.py
│   └── ...
│   
├── ...
├── ...
└── ...
```

### Problem 1

Just run the notebook `problem1.ipynb` in the folder `problem1` (Using the datathon environment).

(For the full dataset prediction part, it is a bit slow.)

### Problem 2

```bash
cd problem2
conda activate datathon
python data_preprocessing.py 
python clustering_model.py 
python pytorch_clustering_models_optimized.py
```

### Problem 3

```bash
cd problem3
conda activate datathon
python credit_line_recommendation.py  # This is a bit slow
```

## Detailed Approach Documentation

### Problem 1: Forecast User Spending for Q4 2025

For this problem, our approach aims to predict customer spending for the fourth quarter of 2025 using historical transaction data. By analyzing users' historical spending patterns and seasonal changes, we build machine learning models to forecast future consumption behavior.

#### Data Processing Pipeline

##### 1. Data Loading and Merging

- Loaded two main transaction datasets: `transaction_fact_20250325.csv` and `wrld_stor_tran_fact_20250325.csv`
- Verified column compatibility between datasets, confirming identical structures
- Merged datasets, resulting in 1,547,190 transaction records

##### 2. Initial Data Analysis

- Examined transaction date range: February 17, 2023 to March 24, 2025
- Identified 14,283 unique accounts
- Analyzed transaction type distribution: SALE, PAYMENT, RETURN, ADJUSTMENT

##### 3. Data Preprocessing

- Selected essential columns: account number, transaction date, transaction amount, transaction type, transaction code
- Saved preprocessed data as `prepared_transactions.csv`

##### 4. Time Series Data Construction

- Added year-month column for monthly aggregation
- Aggregated transaction amounts by account and month
- Created complete time series framework ensuring each account has records for every month
- Filled missing monthly data with 0 (assuming no consumption record means no consumption)
- Added date-related features: year, month, quarter

##### 5. Feature Engineering

- Created time-based features:
  - Lag features (lag_1 to lag_12): Previous 1-12 months spending
  - Moving average features: 3-month, 6-month, 12-month moving averages and standard deviations
  - Year-over-year features: Same month and quarter from previous year
  - Month-over-month and year-over-year growth rates

- Created seasonality features:
  - Month and quarter one-hot encoding
  - Holiday season flag (October-December)
  - Seasonal components (through time series decomposition)

- Created spending pattern features:
  - Recent vs. long-term spending ratios
  - Spending volatility

##### 6. Dataset Splitting

- Training set: March 2023 to September 2024 (271,377 records)
- Validation set: Q4 2024 (October-December, 42,849 records)
- Prediction target: Q4 2025 spending

##### 7. Missing Value Treatment

- Filled lag features, moving averages, and standard deviations with 0
- Filled growth rate features with 0
- Filled other numerical features with median values

#### Model Building and Evaluation

##### 1. Baseline Models

Built two baseline models:
- XGBoost regression model
- Random Forest regression model

##### 2. Model Evaluation Metrics

Used the following metrics to evaluate model performance:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Coefficient of Determination (R²)

##### 3. Model Comparison

Compared the performance of both models on the validation set and selected the best model:
- Random Forest model performed best, with lower RMSE and higher R² values

##### 4. Feature Importance Analysis

Analyzed and visualized the top 15 most important features:
- Recent lag features (lag_1, lag_2, lag_3) were most important
- Short-term moving average (rolling_mean_3) was also significant
- Seasonality features (month indicators) contributed significantly to predictions

#### Prediction Results Visualization

##### 1. Account Spending Time Series

- Randomly selected accounts to display historical spending and predicted spending
- Compared actual values with predicted values during the validation period
- Calculated MAPE for each account

##### 2. Validation Period Detailed Comparison

- Compared actual spending with predicted spending for Q4 2024
- Analyzed error percentages

##### 3. Q4 2025 Predictions

- Used the best model to predict spending for Q4 2025
- Generated individual account predictions and overall spending trends
- Identified high-spending accounts and spending patterns

#### Conclusions and Insights

- Spending patterns show clear seasonality, with Q4 typically being a high-spending season
- Recent months' spending behavior is the strongest indicator for future spending predictions
- The Random Forest model effectively captures spending patterns and provides accurate predictions

#### File Description

- `problem1.ipynb`: Complete Jupyter notebook with data processing and modeling workflows
- `prepared_transactions.csv`: Preprocessed transaction data
- `monthly_spending_time_series.csv`: Monthly aggregated spending time series
- `time_series_features.csv`: Dataset containing all features
- `random_forest_model.pkl`: Saved Random Forest model
- `q4_2025_spending_predictions.csv`: Prediction results for Q4 2025
- Various visualization PNG files 

### Problem 2: Account Classification and Segmentation

#### Project Overview

This project addresses the challenge of classifying credit card accounts into four distinct segments to support strategic decisions regarding credit line management:

1. **Eligible for credit line increase without risk**: Accounts with good credit behavior and stable financial indicators
2. **Eligible for credit line increase with risk**: Accounts that qualify for higher credit limits but present some risk factors
3. **No credit line increase required**: Accounts that have adequate credit lines for their usage patterns
4. **High risk accounts**: Accounts showing signs of delinquency or potential fraud that require monitoring

The classification system helps financial institutions optimize credit line decisions, improve customer satisfaction, and manage risk effectively.

#### Data Processing Workflow

The data processing pipeline (`data_preprocessing.py`) follows these steps:

1. **Data Loading**: Imports 8 distinct data files containing account information, statement details, transaction data, and fraud records
2. **Feature Standardization**: Unifies primary key columns across all tables to `current_account_nbr`
3. **Data Integration**: Joins all tables using left joins based on account numbers
4. **Feature Engineering**: Creates aggregated metrics at the account level including:
   - Balance statistics (mean, max, min, standard deviation)
   - Transaction metrics (amount, frequency, recency)
   - Credit risk indicators (behavior scores, credit bureau scores)
   - Delinquency measures
   - Fraud flags

The preprocessing step produces a unified dataset (`preprocessed_accounts.csv`) with 97,300 accounts and 51 features.

#### Modeling Approach

Two main modeling scripts were developed:

##### 1. K-means Clustering (`clustering_model.py`)

The baseline model uses K-means clustering with the following steps:
- Feature selection and engineering to create credit risk indicators
- Standardization of features to ensure equal weighting
- K-means clustering with 4 clusters
- Silhouette analysis to evaluate clustering quality (score: 0.34)
- Interpretation of clusters based on key financial metrics
- Mapping of technical clusters to business categories based on relative scores

##### 2. Advanced Clustering Methods (`pytorch_clustering_models_optimized.py`)

The advanced approaches implement:

- **Neural Network Autoencoder**: Uses PyTorch to reduce data dimensionality while preserving patterns
- **Memory-Optimized Algorithms**: Implements sampling strategies for handling large datasets
- **Multiple Clustering Methods**:
  - Mini-Batch K-means: Faster version of K-means for large datasets
  - BIRCH: Hierarchical clustering designed for large datasets
  - Hierarchical Clustering: Tree-based approach to identify natural groupings
  - DBSCAN: Density-based clustering that can identify outliers

Each method produces its own classification, which is then mapped to the four business categories.


#### Future Improvements

Potential enhancements for this system include:
1. Incorporating time-series data for trend analysis
2. Developing supervised models based on historical outcomes
3. Implementing ensemble methods to combine strengths of different clustering approaches
4. Creating an automated pipeline for regular classification updates

### Problem 3: Credit Limit Adjustment Recommendations

This project provides credit line increase recommendations for different account segments. The system determines appropriate credit line adjustments for each customer segment based on account risk classification and spending prediction patterns.

#### Data Sources

The system uses two main data sources:
- Risk classification data (`clustered_accounts.csv`): Contains account risk classification results
- Spending prediction data (`q4_2025_spending_predictions.csv`): Contains spending predictions for Q4 2025

#### Methodology

The system combines two methods to generate credit line recommendations:

##### 1. Rule-based Method
- Develops different credit increase strategies based on the risk segment of the account
- Considers risk factors such as behavior scores, credit scores, and credit utilization rates
- Sets different base increase percentages for different segments

##### 2. Machine Learning Method
- Uses Random Forest Regressor to predict optimal credit line increases
- Utilizes features including: credit line, behavior score, credit score, spending prediction, account balance, etc.
- Feature importance analysis shows `cu_crd_line` (current credit line) is the most important feature

#### Visualizations

The system generates the following visualizations:
- Average credit line increase recommendation by account category
- Distribution of recommended increase amounts
- Current credit line vs. recommended increase amount
- Credit line increase percentage by account category

#### Advantages

- Combines rule-based and machine learning methods, balancing expert knowledge and data-driven decisions
- Considers multiple risk factors and spending patterns
- Provides differentiated strategies for customers with different risk levels
- Provides visualization results for easy analysis