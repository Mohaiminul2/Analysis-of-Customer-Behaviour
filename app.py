# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import urllib.request

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    mean_absolute_error, r2_score, silhouette_score
)
import xgboost as xgb
import warnings
import traceback
warnings.filterwarnings('ignore')

# ----- Page config -----
st.set_page_config(
    page_title="Analysis of Customer Behavior",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Cache heavy functions -----
def load_and_clean_data(file_input):
    """Load the raw CSV/Parquet and perform the complete data cleaning pipeline."""
    if file_input.endswith('.parquet'):
        df = pd.read_parquet(file_input)
    else:
        df = pd.read_csv(file_input)
    # Normalize column names (HuggingFace uses Customer_ID, UCI uses Customer ID)
    df.columns = [c.replace('_', ' ') for c in df.columns]
    # Parse dates
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    # Drop rows without Customer ID
    df = df.dropna(subset=['Customer ID'])
    df['Customer ID'] = pd.to_numeric(df['Customer ID'], errors='coerce').astype('Int64')
    # Remove cancelled orders (Invoice starts with 'C')
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    # Remove invalid Quantity or Price
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    # Create TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['Price']
    # Outlier flag (IQR based)
    Q1_qty = df['Quantity'].quantile(0.25)
    Q3_qty = df['Quantity'].quantile(0.75)
    IQR_qty = Q3_qty - Q1_qty
    Q1_price = df['TotalPrice'].quantile(0.25)
    Q3_price = df['TotalPrice'].quantile(0.75)
    IQR_price = Q3_price - Q1_price
    outliers_qty = (df['Quantity'] < Q1_qty - 1.5*IQR_qty) | (df['Quantity'] > Q3_qty + 1.5*IQR_qty)
    outliers_price = (df['TotalPrice'] < Q1_price - 1.5*IQR_price) | (df['TotalPrice'] > Q3_price + 1.5*IQR_price)
    df['is_outlier'] = outliers_qty | outliers_price
    return df

@st.cache_data
def compute_customer_aggregates(df_clean):
    """Create customer-level features."""
    customer_agg = df_clean.groupby('Customer ID').agg({
        'Invoice': 'nunique',
        'TotalPrice': ['sum', 'mean', 'std'],
        'Quantity': 'sum',
        'InvoiceDate': ['min', 'max']
    }).reset_index()
    customer_agg.columns = [
        'Customer ID', 'num_transactions', 'total_spend',
        'avg_order_value', 'std_order_value', 'total_items',
        'first_purchase', 'last_purchase'
    ]
    customer_agg['customer_lifetime_days'] = (
        customer_agg['last_purchase'] - customer_agg['first_purchase']
    ).dt.days
    customer_agg['days_since_last_purchase'] = (
        df_clean['InvoiceDate'].max() - customer_agg['last_purchase']
    ).dt.days
    return customer_agg

@st.cache_data
def compute_cohort_retention(df_clean, customer_agg):
    """Compute cohort retention table."""
    customer_agg = customer_agg.copy()
    df_clean = df_clean.copy()
    customer_agg['cohort_month'] = customer_agg['first_purchase'].dt.to_period('M')
    df_clean['transaction_month'] = df_clean['InvoiceDate'].dt.to_period('M')
    cohort_data = df_clean.merge(
        customer_agg[['Customer ID', 'cohort_month']],
        on='Customer ID'
    )
    cohort_data['cohort_age'] = (
        cohort_data['transaction_month'] - cohort_data['cohort_month']
    ).apply(lambda x: x.n)
    cohort_table = cohort_data.groupby(
        ['cohort_month', 'cohort_age']
    )['Customer ID'].nunique().unstack(fill_value=0)
    cohort_retention = cohort_table.divide(cohort_table.iloc[:, 0], axis=0) * 100
    return cohort_retention

@st.cache_data
def compute_rfm(df_clean):
    """Calculate RFM metrics."""
    observation_date = df_clean['InvoiceDate'].max() + timedelta(days=1)
    rfm = df_clean.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (observation_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalPrice': 'Monetary'
    }).reset_index()
    rfm['Recency_log'] = np.log1p(rfm['Recency'])
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
    return rfm, observation_date

@st.cache_data
def perform_kmeans_clustering(rfm, n_clusters):
    """Run K-Means clustering on log‑scaled RFM and return cluster labels."""
    X = rfm[['Recency_log', 'Frequency_log', 'Monetary_log']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    rfm_result = rfm.copy()
    rfm_result['Cluster'] = labels
    rfm_result['PCA1'] = principal_components[:, 0]
    rfm_result['PCA2'] = principal_components[:, 1]
    return rfm_result, kmeans, scaler, silhouette, pca, X_scaled

def train_churn_model(customer_features):
    """Train XGBoost classifier for churn using base features only (no derived features to avoid multicollinearity)."""
    try:
        # Base features only (no leakage - days_since_last_purchase defines the target!)
        features = customer_features[[
            'num_transactions', 'total_spend', 'customer_lifetime_days'
        ]].copy()
        features = features.fillna(features.median())
        target = customer_features['is_churned'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric='logloss'
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'accuracy': (y_pred == y_test).mean(),
            'report': classification_report(y_test, y_pred, target_names=['Active', 'Churned'])
        }
        return model, scaler, metrics, features.columns.tolist(), (X_test_scaled, y_test, y_proba)
    except Exception as e:
        st.error(f"Churn model training failed: {str(e)}")
        st.text(traceback.format_exc())
        return None, None, None, None, None

def train_ltv_model(customer_features):
    """Train XGBoost regressor for LTV using base features only."""
    try:
        mask = customer_features['future_ltv'] > 0
        features = customer_features[mask][[
            'num_transactions', 'total_spend', 'customer_lifetime_days'
        ]].copy()
        features = features.fillna(features.median())
        target = customer_features[mask]['future_ltv']
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'mean_actual': y_test.mean()
        }
        return model, scaler, metrics, features.columns.tolist()
    except Exception as e:
        st.error(f"LTV model training failed: {str(e)}")
        st.text(traceback.format_exc())
        return None, None, None, None


def compute_segment_profiles(rfm, customer_features):
    """Compute segment profiles with K=5 clustering and churn probabilities."""
    # Clustering with K=5
    rfm_clustered, kmeans_model, scaler_rfm, sil_score, pca_model, _ = perform_kmeans_clustering(rfm, 5)
    
    # Add churn probabilities
    churn_model, churn_scaler, _, churn_feature_names, _ = train_churn_model(customer_features)
    if churn_model is None:
        return None, None
    features = customer_features[churn_feature_names].copy()
    features = features.fillna(features.median())
    scaled_features = churn_scaler.transform(features)
    customer_features = customer_features.copy()
    customer_features['churn_probability'] = churn_model.predict_proba(scaled_features)[:, 1]

    segment_profile = rfm_clustered.merge(
        customer_features[['Customer ID', 'churn_probability']],
        on='Customer ID'
    )

    # Auto-labeling: rank-based approach (consistent with notebook)
    cluster_stats = segment_profile.groupby('Cluster').agg(
        Recency_mean=('Recency', 'mean'),
        Frequency_mean=('Frequency', 'mean'),
        Monetary_mean=('Monetary', 'mean'),
        Count=('Customer ID', 'count')
    )

    # Rank clusters on each dimension (higher = better)
    # For Recency: lower is better, so invert the rank
    cluster_stats['r_score'] = cluster_stats['Recency_mean'].rank(ascending=False)
    cluster_stats['f_score'] = cluster_stats['Frequency_mean'].rank(ascending=True)
    cluster_stats['m_score'] = cluster_stats['Monetary_mean'].rank(ascending=True)
    cluster_stats['composite'] = cluster_stats['r_score'] + cluster_stats['f_score'] + cluster_stats['m_score']

    # Champions = highest composite, Low Value = lowest composite
    label_map = {
        cluster_stats['composite'].idxmax(): 'Champions',
        cluster_stats['composite'].idxmin(): 'Low Value'
    }

    # At Risk: highest recency among remaining
    remaining = [c for c in cluster_stats.index if c not in label_map]
    at_risk_candidates = cluster_stats.loc[remaining].nlargest(1, 'Recency_mean').index
    label_map[at_risk_candidates[0]] = 'At Risk'

    # New Customers: lowest recency among remaining
    remaining = [c for c in cluster_stats.index if c not in label_map]
    new_cust_candidates = cluster_stats.loc[remaining].nsmallest(1, 'Recency_mean').index
    label_map[new_cust_candidates[0]] = 'New Customers'

    # Last remaining gets Core Customers
    remaining = [c for c in cluster_stats.index if c not in label_map]
    label_map[remaining[0]] = 'Core Customers'

    segment_profile['Segment'] = segment_profile['Cluster'].map(label_map)

    # Display profiles
    segment_metrics = segment_profile.groupby('Segment').agg(
        Customers=('Customer ID', 'count'),
        Total_Revenue=('Monetary', 'sum'),
        Avg_Revenue=('Monetary', 'mean'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Churn_Prob=('churn_probability', 'mean')
    ).round(2)
    segment_metrics['Revenue_Share'] = (segment_metrics['Total_Revenue'] / segment_metrics['Total_Revenue'].sum() * 100).round(1)
    segment_metrics['Customer_Share'] = (segment_metrics['Customers'] / segment_metrics['Customers'].sum() * 100).round(1)

    return segment_metrics, segment_profile


# ----- Sidebar -----
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to:",
    ["Data Overview", "Exploratory Analysis", "Cohort Retention",
     "RFM Segmentation", "Churn Model", "LTV Model",
     "Segment Profiles", "Recommendations"]
)

# Using default dataset
default_csv = "data/online_retail_II.csv"
default_parquet = "data/online_retail_II.parquet"
DATASET_URL = "https://huggingface.co/datasets/vancevo/online-retail-ii/resolve/main/data/train-00000-of-00001.parquet"

if os.path.exists(default_csv):
    uploaded_file = default_csv
elif os.path.exists(default_parquet):
    uploaded_file = default_parquet
else:
    # Auto-download from HuggingFace
    os.makedirs("data", exist_ok=True)
    with st.spinner("Dataset not found locally. Downloading from HuggingFace (~7MB)..."):
        try:
            urllib.request.urlretrieve(DATASET_URL, default_parquet)
            uploaded_file = default_parquet
        except Exception as e:
            st.error(f"Failed to download dataset: {e}")
            st.info("Download manually from https://huggingface.co/datasets/vancevo/online-retail-ii and place the parquet file in data/")
            st.stop()

# ----- Load & process data -----
with st.spinner("Loading and cleaning data... This may take a moment."):
    try:
        df_clean = load_and_clean_data(uploaded_file)
        required_columns = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country']
        missing_cols = [c for c in required_columns if c not in df_clean.columns]
        if missing_cols:
            st.error(f"Missing required columns in dataset: {missing_cols}")
            st.stop()

        customer_agg = compute_customer_aggregates(df_clean)
        cohort_retention = compute_cohort_retention(df_clean, customer_agg)
        rfm, observation_date = compute_rfm(df_clean)

        # Prepare customer_features for ML
        customer_features = customer_agg.merge(
            rfm[['Customer ID', 'Recency', 'Frequency', 'Monetary']],
            on='Customer ID', how='left'
        )
        # Churn target
        churn_threshold = 90
        customer_features['is_churned'] = (
            customer_features['days_since_last_purchase'] > churn_threshold
        )
        # LTV target (future 6 months)
        cutoff_date = observation_date - timedelta(days=180)
        future_spend = df_clean[
            df_clean['InvoiceDate'] >= cutoff_date
        ].groupby('Customer ID')['TotalPrice'].sum()
        customer_features = customer_features.merge(
            future_spend.rename('future_ltv'),
            on='Customer ID', how='left'
        )
        customer_features['future_ltv'] = customer_features['future_ltv'].fillna(0)
    except Exception as e:
        st.error(f"Failed to load/process data: {str(e)}")
        st.text(traceback.format_exc())
        st.stop()



# ----- App sections -----
if section == "Data Overview":
    st.header("Data Quality Report")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df_clean):,}")
    col2.metric("Unique Customers", f"{df_clean['Customer ID'].nunique():,}")
    col3.metric("Date Range", f"{df_clean['InvoiceDate'].min():%b %y} – {df_clean['InvoiceDate'].max():%b %y}")
    outlier_pct = df_clean['is_outlier'].sum() / len(df_clean) * 100
    col4.metric("Outliers (%)", f"{outlier_pct:.2f}%")

    st.subheader("Sample Data")
    st.dataframe(df_clean.head(100), use_container_width=True)



elif section == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")

    # Monthly revenue trend
    monthly_revenue = df_clean.groupby(
        df_clean['InvoiceDate'].dt.to_period('M')
    )['TotalPrice'].sum()
    monthly_revenue.index = monthly_revenue.index.to_timestamp()
    fig = px.line(
        x=monthly_revenue.index, y=monthly_revenue.values,
        labels={'x': 'Date', 'y': 'Revenue (£)'},
        title='Monthly Revenue Trend'
    )
    fig.update_traces(line_color='#1f77b4', fill='tozeroy', fillcolor='rgba(31,119,180,0.2)')
    st.plotly_chart(fig, use_container_width=True)

    # Customer distribution plots
    st.subheader("Customer Distributions")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].hist(customer_agg['num_transactions'], bins=50, edgecolor='black', alpha=0.7)
    axes[0,0].set_title('Transactions per Customer')
    axes[0,1].hist(customer_agg['total_spend'], bins=50, edgecolor='black', alpha=0.7)
    axes[0,1].set_title('Total Spend per Customer')
    axes[1,0].hist(customer_agg['avg_order_value'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[1,0].set_title('Average Order Value')
    axes[1,1].hist(customer_agg['customer_lifetime_days'], bins=50, edgecolor='black', alpha=0.7)
    axes[1,1].set_title('Customer Lifetime (days)')
    plt.tight_layout()
    st.pyplot(fig)

    # Top products & countries
    col1, col2 = st.columns(2)
    with col1:
        top_products = df_clean.groupby('Description').agg(
            TotalPrice=('TotalPrice', 'sum'),
            Quantity=('Quantity', 'sum')
        ).nlargest(10, 'TotalPrice')
        fig = px.bar(
            top_products, x='TotalPrice', y=top_products.index,
            orientation='h', title='Top 10 Products by Revenue'
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top_countries = df_clean.groupby('Country').agg(
            TotalPrice=('TotalPrice', 'sum'),
            Customers=('Customer ID', 'nunique')
        ).nlargest(10, 'TotalPrice')
        fig = px.bar(
            top_countries, x='TotalPrice', y=top_countries.index,
            orientation='h', title='Top 10 Countries by Revenue'
        )
        st.plotly_chart(fig, use_container_width=True)

elif section == "Cohort Retention":
    st.header("Cohort Retention Analysis")
    st.markdown("Retention rate = % of customers from a given cohort who made a purchase in a given month after their first purchase.")

    # Pick a subset for heatmap
    max_period = 12
    cohort_subset = cohort_retention.iloc[:min(12, len(cohort_retention)), :max_period+1]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        cohort_subset, annot=True, fmt='.1f', cmap='RdYlGn',
        cbar_kws={'label': 'Retention Rate (%)'},
        vmin=0, vmax=100, ax=ax
    )
    ax.set_title('Cohort Retention Heatmap', fontweight='bold')
    ax.set_xlabel('Months Since First Purchase')
    ax.set_ylabel('Cohort (First Purchase Month)')
    st.pyplot(fig)

elif section == "RFM Segmentation":
    st.header("RFM Customer Segmentation")
    st.markdown("Segment customers based on **Recency**, **Frequency**, and **Monetary** value using K‑Means clustering.")

    # Choose number of clusters
    n_clusters = st.slider("Number of segments (K)", min_value=2, max_value=8, value=5)

    # Run clustering
    with st.spinner("Running K‑Means..."):
        rfm_clustered, kmeans_model, scaler_rfm, sil_score, pca_model, X_scaled = perform_kmeans_clustering(rfm, n_clusters)

    st.subheader(f"Clustering Results (Silhouette Score: {sil_score:.3f})")

    # PCA scatter plot
    fig = px.scatter(
        rfm_clustered, x='PCA1', y='PCA2', color='Cluster',
        size='Monetary', hover_data=['Recency', 'Frequency', 'Monetary'],
        title='Customer Segments (PCA projection)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster profiles
    cluster_profile = rfm_clustered.groupby('Cluster').agg(
        Count=('Customer ID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean')
    ).round(2)
    cluster_profile['Pct_of_customers'] = (cluster_profile['Count'] / len(rfm_clustered) * 100).round(1)
    st.subheader("Cluster Profiles")
    st.dataframe(cluster_profile)

    # Distribution of metrics per cluster
    st.subheader("RFM Distributions by Cluster")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(['Recency', 'Frequency', 'Monetary']):
        for c in rfm_clustered['Cluster'].unique():
            subset = rfm_clustered[rfm_clustered['Cluster'] == c]
            axes[i].hist(subset[metric], bins=30, alpha=0.5, label=f'Cluster {c}')
        axes[i].set_title(metric)
        axes[i].legend()
    st.pyplot(fig)

elif section == "Churn Model":
    st.header("Churn Prediction")

    # Train model
    with st.spinner("Training churn prediction model..."):
        churn_model, churn_scaler, churn_metrics, churn_feature_names, (X_test, y_test, y_proba) = train_churn_model(customer_features)

    if churn_model is None:
        st.stop()

    col1, col2 = st.columns(2)
    col1.metric("ROC‑AUC", f"{churn_metrics['roc_auc']:.3f}")
    col2.metric("Accuracy", f"{churn_metrics['accuracy']:.3f}")

    # Classification report
    st.subheader("Classification Report")
    # Compute predictions from probabilities (threshold 0.5)
    y_pred = (y_proba > 0.5).astype(int)
    # Parse classification report into DataFrame for nice table display
    report_dict = classification_report(y_test, y_pred, target_names=['Active', 'Churned'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    # Format numeric columns
    for col in ['precision', 'recall', 'f1-score']:
        report_df[col] = report_df[col].map(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
    report_df['support'] = report_df['support'].astype(int)
    st.dataframe(report_df, use_container_width=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {churn_metrics["roc_auc"]:.3f})',
        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
        width=500, height=500
    )
    roc_fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(roc_fig, use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': churn_feature_names,
        'Importance': churn_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

elif section == "LTV Model":
    st.header("Lifetime Value Prediction")

    # Train model
    with st.spinner("Training LTV model..."):
        ltv_model, ltv_scaler, ltv_metrics, ltv_feature_names = train_ltv_model(customer_features)

    if ltv_model is None:
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{ltv_metrics['r2']:.3f}")
    col2.metric("Mean Absolute Error", f"£{ltv_metrics['mae']:.2f}")
    col3.metric("Mean Actual LTV", f"£{ltv_metrics['mean_actual']:.2f}")

    # Feature importance
    importance = pd.DataFrame({
        'Feature': ltv_feature_names,
        'Importance': ltv_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

    # Simulate one customer
    st.subheader("Estimate LTV for a Customer")
    with st.form("ltv_form"):
        cola, colb = st.columns(2)
        with cola:
            trans = st.number_input("Transactions", min_value=1, value=10, key="ltv_trans")
            spend = st.number_input("Total spend (£)", min_value=0.0, value=500.0, key="ltv_spend")
        with colb:
            lifetime = st.number_input("Customer lifetime (days)", min_value=1, value=365, key="ltv_life")
            days_last = st.number_input("Days since last purchase", min_value=0, value=30, key="ltv_days_last")

        if st.form_submit_button("Predict LTV"):
            sample = pd.DataFrame([[trans, spend, lifetime, days_last]],
                                  columns=ltv_feature_names)
            sample_scaled = ltv_scaler.transform(sample)
            predicted_ltv = ltv_model.predict(sample_scaled)[0]
            st.success(f"Predicted future LTV (6 months): £{predicted_ltv:,.2f}")

elif section == "Segment Profiles":
    st.header("Segment Profiles")
    with st.spinner("Computing segment profiles..."):
        segment_metrics, segment_profile = compute_segment_profiles(rfm, customer_features)

    if segment_metrics is None:
        st.stop()

    st.dataframe(segment_metrics)

    # Donut charts for segment share
    segment_colors = ['#2563EB', '#14B8A6', '#F97316', '#A855F7', '#E11D48']
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Revenue by Segment', 'Customer Count'),
        specs=[[{'type': 'pie'}, {'type': 'pie'}]]
    )
    fig.add_trace(
        go.Pie(
            labels=segment_metrics.index,
            values=segment_metrics['Total_Revenue'],
            name='Revenue',
            hole=0.58,
            marker=dict(colors=segment_colors, line=dict(color='white', width=3)),
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>Revenue: £%{value:,.2f}<br>Share: %{percent}<extra></extra>',
            pull=[0.04] * len(segment_metrics)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Pie(
            labels=segment_metrics.index,
            values=segment_metrics['Customers'],
            name='Customers',
            hole=0.58,
            marker=dict(colors=segment_colors, line=dict(color='white', width=3)),
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>Customers: %{value:,.0f}<br>Share: %{percent}<extra></extra>',
            pull=[0.04] * len(segment_metrics)
        ),
        row=1, col=2
    )
    fig.update_layout(
        height=520,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.12,
            xanchor='center',
            x=0.5
        ),
        font=dict(size=13),
        margin=dict(t=80, b=80, l=20, r=20),
        annotations=[
            dict(
                text=f"£{segment_metrics['Total_Revenue'].sum():,.0f}<br><span style='font-size:12px'>Total Revenue</span>",
                x=0.19, y=0.5, font=dict(size=18), showarrow=False
            ),
            dict(
                text=f"{segment_metrics['Customers'].sum():,.0f}<br><span style='font-size:12px'>Customers</span>",
                x=0.81, y=0.5, font=dict(size=18), showarrow=False
            )
        ]
    )
    fig.update_traces(
        sort=False,
        direction='clockwise',
        insidetextorientation='radial'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Store segment_metrics in session state for Recommendations page
    st.session_state['segment_metrics'] = segment_metrics

elif section == "Recommendations":
    st.header("Segment-specific Business Recommendations")

    if 'segment_metrics' not in st.session_state:
        st.warning("Please visit the 'Segment Profiles' page first to compute segment metrics.")
        st.stop()

    segment_metrics = st.session_state['segment_metrics']

    for segment in segment_metrics.index:
        row = segment_metrics.loc[segment]
        st.subheader(f"{segment} ({row['Customers']:.0f} customers, {row['Revenue_Share']:.1f}% revenue)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Spend", f"£{row['Avg_Revenue']:,.2f}")
        col2.metric("Avg Recency", f"{row['Avg_Recency']:.0f} days")
        col3.metric("Churn Risk", f"{row['Avg_Churn_Prob']:.0%}")

        if row['Avg_Churn_Prob'] > 0.5:
            st.error("HIGH churn risk - Launch retention campaigns, special discounts.")
        elif row['Avg_Churn_Prob'] > 0.3:
            st.warning("MODERATE risk - Regular engagement, loyalty programme.")
        else:
            st.success("LOW risk - Maintain, upsell, cross-sell.")

        if row['Avg_Revenue'] > segment_metrics['Avg_Revenue'].quantile(0.75):
            st.info("High value - VIP treatment, exclusive offers.")
        if segment == "At Risk":
            st.warning("At-risk but formerly valuable - Win-back campaigns, personalised reach-out.")
        if segment == "New Customers":
            st.info("Recent joiners - Onboarding email series, second-purchase incentive.")
        if segment == "Low Value":
            st.info("Cost-efficient automated nurture only, bundle offers to increase basket size.")
        st.write("---")

# ----- Footer -----
st.sidebar.markdown("---")
st.sidebar.write("Streamlit • UCI Online Retail II")
