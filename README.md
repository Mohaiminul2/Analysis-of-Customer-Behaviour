# Customer Purchase Behaviour Analytics

An interactive customer analytics dashboard for understanding purchase behaviour, finding customer segments, estimating churn risk, and predicting future customer value from Online Retail II transaction data.

The project is designed for two audiences:

| Audience | What this project gives you |
| --- | --- |
| Business and non-technical users | A Streamlit dashboard with plain-language metrics, charts, segment profiles, and recommendations |
| Analysts, data scientists, and students | A reproducible Python workflow covering cleaning, feature engineering, clustering, churn modelling, LTV modelling, and notebook-based analysis |

## What The App Answers

The dashboard focuses on practical customer questions:

1. Which customers are most valuable?
2. Which customer groups behave similarly?
3. Which customers are likely to churn?
4. What future value might a customer generate?
5. Which customer segments deserve retention, VIP, onboarding, or low-cost nurture campaigns?

## Main Features

| Area | What it does |
| --- | --- |
| Data quality | Loads the transaction file, validates required columns, removes cancelled and invalid transactions, and flags outliers |
| Exploratory analysis | Shows revenue trends, customer distributions, top products, and top countries |
| Cohort retention | Shows how monthly customer cohorts return over time |
| RFM segmentation | Groups customers using Recency, Frequency, and Monetary value with K-Means clustering |
| Churn prediction | Trains an XGBoost classifier to estimate whether customers are inactive or at risk |
| LTV prediction | Trains an XGBoost regressor to estimate future 6-month customer value |
| Segment profiles | Creates business-friendly segments such as Champions, Core Customers, At Risk, New Customers, and Low Value |
| Recommendations | Suggests actions for each segment based on value and churn risk |

## Dashboard Pages

The Streamlit app has these sidebar sections:

| Page | Good for |
| --- | --- |
| Data Overview | Checking dataset size, unique customers, date range, outlier rate, and sample rows |
| Exploratory Analysis | Understanding monthly revenue, spending patterns, popular products, and high-revenue countries |
| Cohort Retention | Seeing whether customers keep purchasing after their first order |
| RFM Segmentation | Experimenting with the number of customer clusters and inspecting segment behaviour |
| Churn Model | Reviewing churn model accuracy, ROC-AUC, classification report, and feature importance |
| LTV Model | Reviewing predicted future value and testing a sample customer scenario |
| Segment Profiles | Comparing revenue share and customer share across named segments with donut charts |
| Recommendations | Reading segment-specific business actions |

## How The Pipeline Works

The app follows this flow:

```text
Raw transaction data
        |
        v
Clean cancelled, missing, invalid, and outlier-prone records
        |
        v
Build customer-level features
        |
        v
RFM segmentation + churn model + LTV model
        |
        v
Segment profiles and business recommendations
```

### Data Cleaning

The app loads CSV or Parquet data, then:

- Standardizes column names so both `Customer_ID` and `Customer ID` formats can work.
- Converts `InvoiceDate` to a date/time column.
- Removes rows without `Customer ID`.
- Removes cancelled invoices where the invoice number starts with `C`.
- Removes rows with non-positive quantity or price.
- Creates `TotalPrice = Quantity * Price`.
- Flags quantity and revenue outliers using the IQR method.

### Customer Features

Transactions are aggregated to customer level:

| Feature | Meaning |
| --- | --- |
| `num_transactions` | Number of unique invoices per customer |
| `total_spend` | Total customer revenue |
| `avg_order_value` | Average order value |
| `std_order_value` | Variation in order value |
| `total_items` | Total quantity purchased |
| `first_purchase` | First purchase date |
| `last_purchase` | Most recent purchase date |
| `customer_lifetime_days` | Days between first and last purchase |
| `days_since_last_purchase` | Inactivity period |

### RFM Segmentation

RFM means:

| RFM term | Meaning | Business interpretation |
| --- | --- | --- |
| Recency | Days since last purchase | Lower is usually better |
| Frequency | Number of purchases | Higher often means stronger loyalty |
| Monetary | Total spend | Higher means greater value |

The app log-transforms RFM values, scales them, and uses K-Means clustering. In the RFM page, users can choose `K` from 2 to 8.

For Segment Profiles, the app uses 5 clusters and maps them to business labels:

| Segment | General meaning |
| --- | --- |
| Champions | Best overall RFM behaviour |
| Core Customers | Consistent and valuable customers |
| At Risk | Customers who have not purchased recently |
| New Customers | Recently acquired customers |
| Low Value | Customers with the weakest overall RFM behaviour |

### Churn Model

The churn model is an XGBoost classifier.

| Item | Current app behaviour |
| --- | --- |
| Target | Customer is churned if days since last purchase is greater than 90 |
| Features | Number of transactions, total spend, customer lifetime days |
| Split | 80 percent train, 20 percent test |
| Evaluation | ROC-AUC, accuracy, classification report, ROC curve, feature importance |

### Lifetime Value Model

The LTV model is an XGBoost regressor.

| Item | Current app behaviour |
| --- | --- |
| Target | Future spend from the last 180 days of available data |
| Features | Number of transactions, total spend, customer lifetime days |
| Split | 80 percent train, 20 percent test |
| Evaluation | R2 score, mean absolute error, mean actual LTV, feature importance |

## Dataset

The project expects the Online Retail II transaction dataset.

Required columns:

| Column | Description |
| --- | --- |
| `Invoice` | Invoice number |
| `StockCode` | Product code |
| `Description` | Product description |
| `Quantity` | Quantity purchased |
| `InvoiceDate` | Purchase date and time |
| `Price` | Unit price |
| `Customer ID` | Customer identifier |
| `Country` | Customer country |

The app checks for these columns after loading the data.

### Data Loading Order

The dashboard looks for data in this order:

1. `data/online_retail_II.csv`
2. `data/online_retail_II.parquet`
3. If neither exists, it attempts to download a Parquet dataset from Hugging Face into `data/online_retail_II.parquet`.

This means the dashboard can run without manually downloading the CSV, as long as your internet connection allows the Hugging Face download.

## Project Structure

```text
Analysis of Purchase Behaviour/
├── app.py
├── requirements.txt
├── README.md
├── QUICKSTART.md
├── data/
│   ├── online_retail_II.csv
│   ├── customer_segments.csv
│   └── segment_profiles.csv
├── notebooks/
│   └── 01_customer_behavior_analytics.ipynb
├── models/
└── utils/
```

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard and app-side analytics pipeline |
| `requirements.txt` | Python packages needed to run the dashboard and notebook |
| `data/online_retail_II.csv` | Local raw dataset, if present |
| `data/customer_segments.csv` | Notebook-generated customer-level segment output |
| `data/segment_profiles.csv` | Notebook-generated segment summary output |
| `notebooks/01_customer_behavior_analytics.ipynb` | Longer exploratory and modelling notebook |
| `models/` | Reserved for saved model artifacts |
| `utils/` | Reserved for reusable helper code |

## Installation

See [QUICKSTART.md](QUICKSTART.md) for a shorter setup guide. The usual technical setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Using The Dashboard

After launching the app, open:

```text
http://localhost:8501
```

Suggested path for first-time users:

1. Start with Data Overview to confirm the dataset loaded.
2. Open Exploratory Analysis to understand revenue and customer patterns.
3. Review Cohort Retention to see repeat purchase behaviour.
4. Use RFM Segmentation to inspect customer groups.
5. Open Churn Model to see churn-risk modelling.
6. Open LTV Model to estimate future value.
7. Open Segment Profiles before Recommendations, because Recommendations uses segment metrics generated on the Segment Profiles page.

## Running The Notebook

The notebook gives a deeper, step-by-step analysis. Use it when you want to study the modelling process, modify methodology, or export CSV outputs.

```bash
jupyter notebook notebooks/01_customer_behavior_analytics.ipynb
```

The notebook reads:

```text
data/online_retail_II.csv
```

It can generate:

| Output | Description |
| --- | --- |
| `data/customer_segments.csv` | Customer ID, segment label, churn probability, future LTV, and RFM metrics |
| `data/segment_profiles.csv` | Segment-level summary metrics |

## Customizing The Project

| Goal | Where to change it |
| --- | --- |
| Use a different dataset | Replace `data/online_retail_II.csv` with a file using the required columns |
| Change churn definition | Update `churn_threshold = 90` in `app.py` |
| Change default number of dashboard clusters | Update the RFM slider default in `app.py` |
| Change fixed Segment Profiles cluster count | Update `perform_kmeans_clustering(rfm, 5)` in `compute_segment_profiles()` |
| Add more model features | Update `train_churn_model()` or `train_ltv_model()` |
| Change segment recommendations | Edit the logic in the Recommendations section of `app.py` |

## Important Notes

- The project previously had a local `.venv/` folder. That folder is not source code and can be recreated from `requirements.txt`.
- `data/online_retail_II.csv` is large and is ignored by Git in this working tree. If it is missing, the Streamlit app can attempt to download a Parquet version automatically.
- The notebook currently expects the CSV file specifically, so manual CSV download may still be needed for notebook-only workflows.
- Some model results can vary slightly across package versions and machines.

## Troubleshooting

| Problem | What to try |
| --- | --- |
| `streamlit` command not found | Activate your virtual environment and run `pip install -r requirements.txt` |
| App says required columns are missing | Check that your file has the expected Online Retail II column names |
| App cannot download dataset | Manually place `online_retail_II.csv` or `online_retail_II.parquet` inside `data/` |
| Notebook cannot find data | Place `online_retail_II.csv` in the `data/` folder |
| Port 8501 is already used | Run `streamlit run app.py --server.port 8502` |
| Installation is slow | This is normal; data science packages such as SciPy, scikit-learn, XGBoost, and Jupyter are large |
| Folder becomes very large | Delete and recreate `.venv/` when needed; keep source files separate from local environments |

## Technologies Used

| Category | Tools |
| --- | --- |
| App | Streamlit |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Visualization | Plotly, Matplotlib, Seaborn |
| Notebook workflow | Jupyter, JupyterLab, IPython |

## Who Can Use This

Non-technical users can run the dashboard and read the charts without touching the notebook.

Technical users can inspect `app.py`, modify model logic, run the notebook, and adapt the workflow to a different retail dataset.

## License

No license file is currently included in this working tree. Add one before publishing if you want to clarify reuse permissions.
