# Quick Start

This guide gets the Customer Purchase Behaviour Analytics dashboard running as quickly as possible.

Use the README when you want the full project explanation. Use this file when you just want to run it.

## What You Need

| Requirement | Why it is needed |
| --- | --- |
| Python 3.10 or newer | Runs the dashboard and analysis code |
| Internet connection | Needed for package installation and optional dataset auto-download |
| 1 GB free disk space | Enough for Python packages, dataset, and temporary files |

## For Non-Technical Users

These steps are written for someone who just wants to open the dashboard.

### Step 1: Open A Terminal

On macOS, open Terminal.

On Windows, open PowerShell.

### Step 2: Go To The Project Folder

If you already have this project folder open locally, go into it:

```bash
cd "/path/to/Analysis of Purchase Behaviour"
```

On this machine, the current project path is:

```bash
cd "/Users/mohaiminulislam/_MyWrkSpace/2.WIP/Analysis of Purchase Behaviour"
```

### Step 3: Create A Fresh Python Environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

After activation, your terminal usually shows `(.venv)` at the beginning of the line.

### Step 4: Install The Required Packages

```bash
pip install -r requirements.txt
```

This can take a few minutes because the project uses data science and dashboard libraries.

### Step 5: Start The Dashboard

```bash
streamlit run app.py
```

The dashboard should open automatically in your browser. If it does not, open:

```text
http://localhost:8501
```

### Step 6: Let The Data Load

The app loads data in this order:

1. It first checks for `data/online_retail_II.csv`.
2. If that is missing, it checks for `data/online_retail_II.parquet`.
3. If both are missing, it tries to download a Parquet version automatically.

The first run may take longer because data has to be loaded, cleaned, and cached.

## What To Click First

Recommended first tour:

1. Data Overview
2. Exploratory Analysis
3. Cohort Retention
4. RFM Segmentation
5. Churn Model
6. LTV Model
7. Segment Profiles
8. Recommendations

Important: open Segment Profiles before Recommendations. The Recommendations page uses segment metrics created on the Segment Profiles page.

## For Technical Users

### Standard Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Run On A Different Port

Use this if port `8501` is already busy:

```bash
streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

### Run The Notebook

The notebook is useful if you want to inspect the full analysis workflow and generate CSV outputs.

```bash
jupyter notebook notebooks/01_customer_behavior_analytics.ipynb
```

The notebook expects:

```text
data/online_retail_II.csv
```

If you only have the automatically downloaded Parquet file, use the Streamlit app directly or manually download the CSV before running the notebook.

## Dataset Options

### Option A: Use The Existing Local CSV

If this file exists, you are ready:

```text
data/online_retail_II.csv
```

### Option B: Let The App Auto-Download

If no local CSV or Parquet file exists, the app attempts to download:

```text
data/online_retail_II.parquet
```

This is the easiest option for dashboard users.

### Option C: Manually Add The Dataset

Place either of these files inside `data/`:

```text
online_retail_II.csv
online_retail_II.parquet
```

The file should contain these columns:

```text
Invoice
StockCode
Description
Quantity
InvoiceDate
Price
Customer ID
Country
```

## Common Problems

| Problem | Fix |
| --- | --- |
| `python3` not found | Try `python` instead |
| `streamlit` not found | Activate `.venv`, then run `pip install -r requirements.txt` |
| Browser does not open | Manually open `http://localhost:8501` |
| Dataset download fails | Add `online_retail_II.csv` or `online_retail_II.parquet` to `data/` manually |
| Notebook cannot find data | Make sure `data/online_retail_II.csv` exists |
| Install takes a long time | Wait; packages like SciPy, XGBoost, Jupyter, and Streamlit are large |
| Project folder gets too large | The `.venv/` folder can be deleted and recreated later |

## Stop The Dashboard

Go back to the terminal where Streamlit is running and press:

```text
Ctrl + C
```

## Restart Later

If `.venv/` already exists:

macOS or Linux:

```bash
source .venv/bin/activate
streamlit run app.py
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
streamlit run app.py
```

If `.venv/` was deleted, recreate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Quick Explanation Of The Dashboard

| Page | Meaning |
| --- | --- |
| Data Overview | Confirms the dataset loaded and shows basic quality metrics |
| Exploratory Analysis | Shows revenue, customer, product, and country patterns |
| Cohort Retention | Shows how often customer groups return after first purchase |
| RFM Segmentation | Groups customers by recency, frequency, and spending |
| Churn Model | Estimates customers likely to become inactive |
| LTV Model | Estimates future customer value |
| Segment Profiles | Shows named customer segments and their revenue/customer share |
| Recommendations | Suggests actions for each customer segment |

## Next Step

After the dashboard is running, read [README.md](README.md) for the full explanation of the project, methodology, folder structure, and customization options.
