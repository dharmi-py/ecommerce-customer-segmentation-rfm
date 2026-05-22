E-Commerce Customer Segmentation using RFM and Clustering
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.1+-orange.svg)
Project Overview
This project performs customer segmentation on e-commerce transaction data using RFM (Recency, Frequency, Monetary) analysis combined with K-Means clustering. The goal is to identify distinct customer groups to enable targeted marketing campaigns, improve retention, and maximize customer lifetime value.
Business Problem
An e-commerce company wants to:
Understand its customer base beyond simple demographics
Design targeted marketing campaigns for different customer types
Identify high-value customers for retention programs
Re-engage inactive or at-risk customers
Optimize marketing spend by focusing on segments with highest ROI
Dataset Description
Column	Description
`InvoiceNo`	Unique invoice identifier (C-prefix = cancelled)
`StockCode`	Product SKU
`Description`	Product name
`Category`	Product category (Electronics, Apparel, Home, Beauty, etc.)
`Quantity`	Units purchased
`InvoiceDate`	Transaction timestamp
`UnitPrice`	Price per unit
`CustomerID`	Unique customer identifier
`Country`	Customer location
Source: Synthetic dataset for academic use
Period: January 2025 - November 2025
Transactions: ~1,000 rows (after cleaning: ~750 rows)
Customers: ~200 unique customers
Countries: UK, India, Germany, France, Netherlands, Spain, Ireland, Portugal, Singapore, Canada, Australia
Data Cleaning Summary
Step	Action	Rows Removed
1	Convert InvoiceDate to datetime	0
2	Remove missing CustomerID	92
3	Remove cancelled invoices (C-prefix)	28
4	Remove negative/zero quantities	15
5	Remove zero/negative unit prices	12
6	Remove duplicate records	5
7	Fill missing descriptions via StockCode mapping	0
Total		~150 rows (15%)
Rationale: Missing CustomerIDs prevent customer-level analysis. Cancelled orders and negative values represent returns/errors. Duplicates inflate metrics artificially.
Feature Engineering Summary
Customer-Level Features
TotalRevenue: Lifetime spend per customer
PurchaseFrequency: Number of unique orders
TotalQuantity: Total units purchased
UniqueProducts: Product variety breadth (cross-category engagement)
AvgOrderValue: Mean spend per transaction
Recency: Days since last purchase (from reference date: 2025-11-28)
Tenure: Days between first and last purchase
Country: Customer's primary country
RFM Table
Feature	Description	Business Meaning
Recency	Days since last purchase	Lower = more recent = more engaged
Frequency	Number of purchases	Higher = more loyal
Monetary	Total revenue	Higher = more valuable
EDA Insights
Geographic: UK dominates sales (~50%), with India, Germany, and France as secondary markets
Product Performance: Running Shoes, Denim Jacket, and Bluetooth Speaker are top revenue drivers
Purchase Behavior: Most customers are one-time or occasional buyers; loyalty is concentrated in a small group
Outliers: Bulk buyers and premium product purchasers represent high-value opportunities
Seasonality: Relatively stable sales throughout 2025 with minor monthly fluctuations
RFM Correlation: Frequency and Monetary strongly correlated (0.62) — frequent buyers spend more
Clustering Approach
Methodology
Algorithm: K-Means Clustering
Features: Recency, log(Frequency), log(Monetary)
Scaling: StandardScaler (Z-score normalization)
Optimal K: 4 (selected via Elbow Method + Silhouette Score + Business Interpretability)
Silhouette Score: 0.31
Why K=4?
Elbow shows clear bend at K=4
Silhouette score peaks at K=4
4 clusters map cleanly to actionable business segments
Cluster Interpretation
Cluster	Name	Size	Avg Recency	Avg Frequency	Avg Revenue	Revenue Share	Profile
0	Recent Low-Value Buyers	25.5%	67 days	1.7	~10,000	19.2%	Recently purchased but low frequency/spend
1	At-Risk Moderate Buyers	24.5%	232 days	1.6	~11,300	20.8%	Moderate spenders, haven't bought in ~8 months
2	High-Value Loyal Customers	25.5%	74 days	3.4	~26,000	47.7%	Frequent, recent, high spenders
3	Lost/Lapsed Customers	24.5%	166 days	1.0	~1,700	12.3%	Single purchase, long inactive, low value
Detailed Profiles
Cluster 0 — Recent Low-Value Buyers
Behavior: Recently discovered the store, made 1-2 purchases
Spending: Low to moderate (~10,000 per customer)
Value: Growth potential; need nurturing
Cluster 1 — At-Risk Moderate Buyers
Behavior: Previously engaged, showing declining activity
Spending: Moderate (~11,300 per customer)
Value: High reactivation potential
Cluster 2 — High-Value Loyal Customers
Behavior: Frequent buyers with high spending and recent activity
Spending: Very high (~26,000 per customer)
Value: Most valuable segment; retain at all costs
Cluster 3 — Lost/Lapsed Customers
Behavior: One-time buyers who never returned
Spending: Very low (~1,700 per customer)
Value: Low immediate value; understand churn reasons
Business Recommendations
Cluster 0: Recent Low-Value Buyers (25.5%)
Goal: Convert to repeat buyers
Send personalized welcome email series with product recommendations
Offer first-repeat purchase discount (10-15%)
Use retargeting ads for browsed/abandoned products
Introduce loyalty program early to build purchase habit
Cluster 1: At-Risk Moderate Buyers (24.5%)
Goal: Win back before they churn
Launch "We miss you" email campaign with time-limited offer (15-20% off)
Send survey to understand reasons for absence
Recommend new arrivals based on past purchase categories
Consider personalized reactivation call for highest-value at-risk customers
Cluster 2: High-Value Loyal Customers (25.5%)
Goal: Retain and maximize lifetime value
Create VIP loyalty tier with exclusive perks (free shipping, early access, birthday gifts)
Assign personal account manager for top 10% spenders
Launch referral program with meaningful rewards
Cross-sell premium products and curated bundles
Maintain priority customer service channel
Cluster 3: Lost/Lapsed Customers (24.5%)
Goal: Low-cost reactivation or clean list
Send one final "Come back" email with strong offer (20-25% off)
Include exit survey to understand churn drivers
If no response, reduce marketing frequency to minimize costs
Focus resources on higher-value segments instead
Country-Specific Strategies
UK: Largest market — invest in loyalty programs and premium offerings
India/Germany/France: Secondary markets — localized campaigns and currency/pricing optimization
Other countries: Evaluate expansion potential vs. current performance
How to Run the Project
Prerequisites
```bash
# Python 3.9 or higher
python --version

# Install dependencies
pip install -r requirements.txt
```
Option 1: Run the Python Script
```bash
# Place dataset in the same directory
python main.py
```
Option 2: Run the Jupyter Notebook
```bash
jupyter notebook notebook.ipynb
```
Output Files
`customer_segments.csv` — Customer-level features with cluster assignments
`images/` — All EDA and clustering visualizations
Project Structure
```
.
├── README.md                          # This file
├── main.py                            # Executable Python script
├── notebook.ipynb                     # Jupyter notebook version
├── requirements.txt                   # Python dependencies
├── dataset_source.md                  # Dataset documentation
├── part_1_ecommerce_customer_segmentation.csv  # Dataset
├── customer_segments.csv              # Output: Segmented customers
└── images/                            # Output: Visualizations
    ├── 01_sales_by_country.png
    ├── 02_top_products_quantity.png
    ├── 03_top_products_revenue.png
    ├── 04_purchase_frequency_dist.png
    ├── 05_avg_order_value_dist.png
    ├── 06_outlier_analysis.png
    ├── 07_high_value_customers.png
    ├── 08_monthly_sales_trend.png
    ├── 09_rfm_correlation.png
    ├── 10_elbow_method.png
    ├── 11_cluster_visualization.png
    ├── 12_cluster_comparison.png
    └── 13_cluster_country.png
```
Dependencies
Package	Version	Purpose
pandas	>=1.5.0	Data manipulation
numpy	>=1.21.0	Numerical computing
matplotlib	>=3.5.0	Visualization
seaborn	>=0.12.0	Statistical visualization
scikit-learn	>=1.1.0	K-Means clustering
jupyter	>=1.0.0	Notebook environment
Author
Business Analytics Project — Part 1: E-Commerce Customer Segmentation
License
This project is for academic purposes. The dataset is synthetic and does not represent real customer data.
