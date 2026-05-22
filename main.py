"""
E-Commerce Customer Segmentation using RFM and K-Means Clustering
================================================================
Business Analytics Project - Part 1

This script performs:
1. Data loading and understanding
2. Data cleaning (missing values, cancellations, outliers, duplicates)
3. Feature engineering (customer-level features, RFM table)
4. Exploratory Data Analysis with visualizations
5. K-Means clustering on RFM features
6. Cluster interpretation and business recommendations

Usage:
    python main.py

Requirements:
    pip install -r requirements.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'part_1_ecommerce_customer_segmentation.csv'
OUTPUT_DIR = 'images'
RANDOM_STATE = 42
OPTIMAL_K = 4


def load_data(path):
    """Load and perform initial data understanding."""
    print("=" * 60)
    print("1. DATA UNDERSTANDING")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing Values:
{df.isnull().sum()}")
    print(f"Duplicated Rows: {df.duplicated().sum()}")
    print(f"Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    return df


def clean_data(df):
    """Clean the dataset for analysis."""
    print("
" + "=" * 60)
    print("2. DATA CLEANING")
    print("=" * 60)

    original_count = len(df)

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print("1. Converted InvoiceDate to datetime")

    # Remove rows with missing CustomerID
    df = df[df['CustomerID'].notna()].copy()
    print(f"2. Removed {original_count - len(df)} rows with missing CustomerID")

    # Remove cancelled invoices (start with 'C')
    df = df[~df['InvoiceNo'].str.startswith('C')].copy()
    print("3. Removed cancelled invoices")

    # Remove negative or zero quantities
    df = df[df['Quantity'] > 0].copy()
    print("4. Removed negative/zero quantities")

    # Remove zero or negative unit prices
    df = df[df['UnitPrice'] > 0].copy()
    print("5. Removed zero/negative unit prices")

    # Remove duplicates
    df = df.drop_duplicates().copy()
    print("6. Removed duplicate rows")

    # Fill missing descriptions using StockCode mapping
    desc_map = df.dropna(subset=['Description']).drop_duplicates('StockCode').set_index('StockCode')['Description']
    df['Description'] = df.apply(
        lambda row: desc_map.get(row['StockCode'], row['Description']) if pd.isna(row['Description']) else row['Description'],
        axis=1
    )
    print("7. Filled missing descriptions via StockCode mapping")

    # Create Revenue column
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    print(f"
Cleaning Summary: {original_count} -> {len(df)} rows ({(original_count - len(df))/original_count*100:.1f}% removed)")
    return df


def engineer_features(df):
    """Create customer-level features and RFM table."""
    print("
" + "=" * 60)
    print("3. FEATURE ENGINEERING")
    print("=" * 60)

    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    print(f"Reference Date: {reference_date.strftime('%Y-%m-%d')}")

    # Customer-level aggregation
    customer_features = df.groupby('CustomerID').agg({
        'Revenue': 'sum',
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'StockCode': 'nunique',
        'InvoiceDate': ['min', 'max'],
        'Country': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
    }).reset_index()

    customer_features.columns = ['CustomerID', 'TotalRevenue', 'PurchaseFrequency',
                                  'TotalQuantity', 'UniqueProducts', 'FirstPurchase',
                                  'LastPurchase', 'Country']

    # Derived features
    customer_features['Recency'] = (reference_date - customer_features['LastPurchase']).dt.days
    customer_features['AvgOrderValue'] = customer_features['TotalRevenue'] / customer_features['PurchaseFrequency']
    customer_features['Tenure'] = (customer_features['LastPurchase'] - customer_features['FirstPurchase']).dt.days

    # RFM table
    rfm = customer_features[['CustomerID', 'Recency', 'PurchaseFrequency', 'TotalRevenue']].copy()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    print(f"Created features for {len(customer_features)} customers")
    return customer_features, rfm


def perform_eda(df, customer_features, output_dir):
    """Generate EDA visualizations."""
    print("
" + "=" * 60)
    print("4. EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Sales by Country
    plt.figure(figsize=(10, 5))
    country_sales = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False)
    country_sales.plot(kind='bar', color='steelblue')
    plt.title('Total Sales by Country')
    plt.xlabel('Country')
    plt.ylabel('Total Revenue')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sales_by_country.png', dpi=150)
    plt.close()

    # 2. Top Products by Revenue
    plt.figure(figsize=(10, 5))
    df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(15).plot(kind='barh', color='mediumseagreen')
    plt.title('Top 15 Products by Revenue')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_products_revenue.png', dpi=150)
    plt.close()

    # 3. Purchase Frequency Distribution
    plt.figure(figsize=(10, 5))
    customer_features['PurchaseFrequency'].hist(bins=20, color='mediumpurple', edgecolor='white')
    plt.title('Distribution of Customer Purchase Frequency')
    plt.xlabel('Number of Purchases')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/purchase_frequency_dist.png', dpi=150)
    plt.close()

    # 4. RFM Correlation
    plt.figure(figsize=(8, 6))
    rfm_corr = customer_features[['Recency', 'PurchaseFrequency', 'TotalRevenue']].corr()
    rfm_corr.columns = ['Recency', 'Frequency', 'Monetary']
    rfm_corr.index = ['Recency', 'Frequency', 'Monetary']
    sns.heatmap(rfm_corr, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f')
    plt.title('RFM Features Correlation')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rfm_correlation.png', dpi=150)
    plt.close()

    print(f"EDA charts saved to {output_dir}/")


def perform_clustering(rfm, customer_features, output_dir):
    """Perform K-Means clustering on RFM features."""
    print("
" + "=" * 60)
    print("5. CUSTOMER SEGMENTATION (K-MEANS)")
    print("=" * 60)

    # Prepare features
    features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    features['Frequency_log'] = np.log1p(features['Frequency'])
    features['Monetary_log'] = np.log1p(features['Monetary'])
    X = features[['Recency', 'Frequency_log', 'Monetary_log']].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    inertias = []
    silhouettes = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))

    # Plot elbow
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(2, 11), inertias, 'bo-')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Inertia')

    ax2.plot(range(2, 11), silhouettes, 'ro-')
    ax2.set_title('Silhouette Score')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Silhouette')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/elbow_method.png', dpi=150)
    plt.close()

    # Final model
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    rfm['Cluster'] = labels
    customer_features['Cluster'] = labels

    print(f"Optimal K = {OPTIMAL_K}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.3f}")
    print(f"Cluster Distribution: {dict(pd.Series(labels).value_counts().sort_index())}")

    return rfm, customer_features


def interpret_clusters(customer_features):
    """Interpret each cluster and provide business recommendations."""
    print("
" + "=" * 60)
    print("6. CLUSTER INTERPRETATION & RECOMMENDATIONS")
    print("=" * 60)

    cluster_names = {
        0: 'Recent Low-Value Buyers',
        1: 'At-Risk Moderate Buyers',
        2: 'High-Value Loyal Customers',
        3: 'Lost/Lapsed Customers'
    }

    for cid in range(OPTIMAL_K):
        data = customer_features[customer_features['Cluster'] == cid]
        revenue_share = data['TotalRevenue'].sum() / customer_features['TotalRevenue'].sum() * 100

        print(f"
--- Cluster {cid}: {cluster_names[cid]} ---")
        print(f"  Size: {len(data)} customers ({len(data)/len(customer_features)*100:.1f}%)")
        print(f"  Avg Recency: {data['Recency'].mean():.1f} days")
        print(f"  Avg Frequency: {data['PurchaseFrequency'].mean():.2f}")
        print(f"  Avg Revenue: {data['TotalRevenue'].mean():.2f}")
        print(f"  Revenue Share: {revenue_share:.1f}%")

    print("
" + "=" * 60)
    print("BUSINESS RECOMMENDATIONS")
    print("=" * 60)

    recommendations = """
    Cluster 0 - Recent Low-Value Buyers:
    - Send welcome series emails with product recommendations
    - Offer first-purchase discount to encourage second order
    - Use retargeting ads for products they viewed
    - Introduce loyalty program early to build habit

    Cluster 1 - At-Risk Moderate Buyers:
    - Launch win-back email campaign with "We miss you" messaging
    - Offer time-limited discount (15-20%) to re-engage
    - Survey to understand reasons for absence
    - Recommend new arrivals based on past purchases

    Cluster 2 - High-Value Loyal Customers:
    - VIP loyalty program with exclusive perks and early access
    - Personal account manager for top spenders
    - Referral incentives to leverage advocacy
    - Cross-sell premium products and bundles
    - Maintain excellent customer service priority

    Cluster 3 - Lost/Lapsed Customers:
    - Low-cost reactivation email with strong offer
    - Analyze exit reasons via survey
    - Focus resources on higher-value segments
    - Consider removing from regular marketing to reduce costs
    """
    print(recommendations)


def main():
    """Main execution pipeline."""
    # Load data
    df = load_data(DATA_PATH)

    # Clean data
    df_clean = clean_data(df)

    # Engineer features
    customer_features, rfm = engineer_features(df_clean)

    # EDA
    perform_eda(df_clean, customer_features, OUTPUT_DIR)

    # Clustering
    rfm, customer_features = perform_clustering(rfm, customer_features, OUTPUT_DIR)

    # Interpretation
    interpret_clusters(customer_features)

    # Save results
    customer_features.to_csv('customer_segments.csv', index=False)
    print("
Customer segments saved to customer_segments.csv")
    print("Project complete!")


if __name__ == '__main__':
    main()
