# Dataset Source

## Source Information
- **Dataset**: `part_1_ecommerce_customer_segmentation.csv`
- **Origin**: Synthetic dataset provided as part of the Business Analytics and Machine Learning academic project.
- **Purpose**: Customer segmentation using RFM analysis and K-Means clustering.

## Dataset Description
This dataset contains transaction-level e-commerce data designed to simulate realistic online retail operations. It includes intentional data quality issues for educational purposes in data cleaning and preprocessing.

## Columns
| Column | Description |
|--------|-------------|
| InvoiceNo | Unique invoice identifier (prefixed with 'C' for cancelled orders) |
| StockCode | Product stock code / SKU |
| Description | Product description |
| Category | Product category (e.g., Electronics, Apparel, Home) |
| Quantity | Number of units purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per unit |
| CustomerID | Unique customer identifier |
| Country | Customer's country |

## Data Quality Issues (Intentional)
- Missing CustomerID values
- Missing product descriptions
- Cancelled invoices (negative quantities, prefixed with 'C')
- Zero and negative unit prices
- Duplicate records
- Incorrect data types

## Usage Rights
This dataset is provided for academic and educational use only. It is synthetic and does not represent any real business or customer data.
