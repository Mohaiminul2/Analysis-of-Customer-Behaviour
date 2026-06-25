# Analysis-of-Customer-Behaviour

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

## Running The Notebook

The notebook gives a deeper, step-by-step analysis. Use it when you want to study the modelling process, modify methodology, or export CSV outputs.

```bash
jupyter notebook notebooks/01_customer_behavior_analytics.ipynb
```

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
