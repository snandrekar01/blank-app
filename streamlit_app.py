# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote
from datetime import datetime
import hashlib
import plotly.express as px
from vaderSentiment.vaderSentiment im        except Exception as e:
            st.session_state["last_prediction"] = {"error": f"{type(e).__name__}: {e}"}
            st.error(
                "❌ Prediction failed.\n\n"
                f"{type(e).__name__}: {e}\n\n"
                "Check your BigQuery connection and permissions."
            )

# Show last prediction if available
if "last_prediction" in st.session_state and st.session_state["last_prediction"]:
    info = st.session_state["last_prediction"]
    if "error" in info:
        with st.expander("Last Prediction Error"):
            st.code(info["error"])
    elif not predict_clicked:  # Only show if we're not already showing new results
        with st.expander("Last Prediction Results"):
            sentiment_color = "green" if info["sentiment"] > 0 else "red"
            pred_color = "green" if info["prediction"] > 0 else "red"
            st.markdown(f"**Sentiment score:** ::{sentiment_color}[{info['sentiment']:.3f}]")
            st.markdown(f"**Predicted return:** ::{pred_color}[{info['prediction']:.2%}]")
            st.text(info["model"].summary())sityAnalyzer
import statsmodels.api as sm

from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account

YF_BASE = "https://finance.yahoo.com"
DEFAULT_SYMBOL = "^GSPC"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

st.set_page_config(page_title="Yahoo Finance News Collector", page_icon="📰", layout="centered")

# ---- helpers ----------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_yahoo_news(symbol: str) -> pd.DataFrame:
    sym_enc = quote(symbol, safe="")
    url = f"{YF_BASE}/quote/{sym_enc}/news"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} fetching {url}")
    soup = BeautifulSoup(resp.text, "html.parser")

    items = []
    for h3 in soup.find_all("h3"):
        a = h3.find("a", href=True) or h3.find_parent("a", href=True)
        if not a:
            continue
        title = h3.get_text(strip=True)
        href = a["href"].strip()
        if not href:
            continue
        if href.startswith("/"):
            href = urljoin(YF_BASE, href)
        if "yahoo.com" in href and "/news" not in href:
            continue
        if title:
            items.append({"title": title, "url": href})

    # dedupe by URL
    seen, unique = set(), []
    for it in items:
        if it["url"] in seen:
            continue
        seen.add(it["url"])
        unique.append(it)

    df = pd.DataFrame(unique)
    if df.empty:
        return df
    df.insert(0, "symbol", symbol)
    df["fetched_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return df

def get_bq_client(project_id_input: str | None) -> bigquery.Client:
    if "gcp_service_account" in st.secrets:
        sa_info = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project = project_id_input or sa_info.get("project_id")
        if not project:
            raise ValueError("Project ID is required (sidebar or secrets)")
        return bigquery.Client(project=project, credentials=creds)
    if not project_id_input:
        raise ValueError("Project ID is required when using ADC")
    return bigquery.Client(project=project_id_input)

def ensure_dataset_and_table(client: bigquery.Client, dataset_id: str, table_id: str, location: str = "EU") -> str:
    ds_ref = bigquery.Dataset(f"{client.project}.{dataset_id}")
    ds_ref.location = location or "EU"
    try:
        ds = client.get_dataset(ds_ref)
        if ds.location.upper() != (location or "EU").upper():
            raise RuntimeError(
                f"Dataset '{dataset_id}' exists in {ds.location}, but app selected {location}. "
                "Change the sidebar 'Location' to match."
            )
    except NotFound:
        client.create_dataset(ds_ref)

    full_table_id = f"{client.project}.{dataset_id}.{table_id}"
    try:
        client.get_table(full_table_id)
    except NotFound:
        schema = [
            bigquery.SchemaField("symbol", "STRING"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("url", "STRING"),
            bigquery.SchemaField("fetched_at", "TIMESTAMP"),
            bigquery.SchemaField("url_id", "STRING"),
            bigquery.SchemaField("added_at", "TIMESTAMP"),
        ]
        table = bigquery.Table(full_table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(field="fetched_at", type_=bigquery.TimePartitioningType.DAY)
        client.create_table(table)
        st.success(f"✓ Table created: {table_id}")
    return full_table_id

def save_to_bigquery(client: bigquery.Client, df: pd.DataFrame, dataset_id: str, table_id: str, location: str) -> dict:
    # ensure structure FIRST so first run works
    full_target = ensure_dataset_and_table(client, dataset_id, table_id, location)

    if df.empty:
        return {"attempted": 0, "new_rows": 0, "target": full_target}

    df2 = df.copy()
    df2["url_id"] = df2["url"].apply(lambda u: hashlib.md5(u.encode("utf-8")).hexdigest())
    df2["fetched_at"] = pd.to_datetime(df2["fetched_at"], utc=True)
    df2["added_at"] = pd.Timestamp.now(tz="UTC")

    staging = f"{full_target}__staging"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(df2, staging, job_config=job_config).result()

    pre_sql = f"""
      SELECT COUNT(1) AS to_insert
      FROM `{staging}` S
      LEFT JOIN `{full_target}` T
        ON T.url_id = S.url_id
      WHERE T.url_id IS NULL
    """
    pre_res = list(client.query(pre_sql).result())
    to_insert = int(pre_res[0]["to_insert"]) if pre_res else 0

    merge_sql = f"""
    MERGE `{full_target}` T
    USING `{staging}` S
      ON T.url_id = S.url_id
    WHEN NOT MATCHED THEN
      INSERT (symbol, title, url, fetched_at, url_id, added_at)
      VALUES (S.symbol, S.title, S.url, S.fetched_at, S.url_id, S.added_at)
    """
    client.query(merge_sql).result()
    client.delete_table(staging, not_found_ok=True)

    return {"attempted": int(len(df2)), "new_rows": to_insert, "target": full_target}

def calculate_sentiment_scores(headlines_df: pd.DataFrame) -> float:
    """Calculate average compound sentiment score for headlines."""
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    
    for title in headlines_df['title']:
        sentiment = analyzer.polarity_scores(title)
        scores.append(sentiment['compound'])
    
    return pd.Series(scores).mean() if scores else 0.0

def fetch_historical_data(client: bigquery.Client) -> pd.DataFrame:
    """Fetch historical price and sentiment data from BigQuery."""
    query = """
    SELECT 
        Date,
        daily_return_pct as return,
        aggregated_daily_compound_avg,
        LAG(aggregated_daily_compound_avg, 1) OVER(ORDER BY Date) as aggregated_daily_compound_avg_lag
    FROM `project-market-news.Final_data_do_not_touch.final_merged_table`
    ORDER BY Date
    """
    return client.query(query).to_dataframe()

# ---- app --------------------------------------------------------------------

# keep state across reruns
if "news_df" not in st.session_state:
    st.session_state["news_df"] = None
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

st.title("🗞️ Yahoo Finance News Collector")
st.caption("Fetch Yahoo Finance headlines and optionally append them to BigQuery (deduped by URL).")

fetch_clicked = st.button("Fetch S&P 500 News", type="primary")

# Sidebar config
st.sidebar.header("BigQuery (optional)")
save_bq = st.sidebar.checkbox("Enable BigQuery save", value=True)
project_id = st.sidebar.text_input("GCP Project ID", value=st.secrets.get("gcp_project_id", ""))
dataset_id = st.sidebar.text_input("Dataset", value="market_news")
table_id = st.sidebar.text_input("Table", value="yahoo_headlines")
location = st.sidebar.selectbox("Location", ["EU", "US"], index=0, help="Dataset location")

# On fetch, persist the df into session so it survives reruns
if fetch_clicked:
    try:
        df_fetched = fetch_yahoo_news(DEFAULT_SYMBOL)
        if df_fetched.empty:
            st.session_state["news_df"] = None
            st.info("No headlines found.")
        else:
            st.session_state["news_df"] = df_fetched
            st.success(f"Fetched {len(df_fetched)} headlines from S&P 500.")
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")

# Show data if available in session
df = st.session_state["news_df"]
if isinstance(df, pd.DataFrame) and not df.empty:
    st.subheader("Headlines")
    for _, row in df.iterrows():
        st.markdown(f"- [{row['title']}]({row['url']})")

    st.subheader("Data")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"SP500_news_{datetime.utcnow().date()}.csv",
        mime="text/csv",
    )

# Prediction section is shown when we have news data
if isinstance(df, pd.DataFrame) and not df.empty:
    st.subheader("Sentiment Analysis & Predictions")
    st.info("Click to analyze sentiment and predict tomorrow's returns")
    predict_clicked = st.button("Get Predictions for Tomorrow", type="primary")

    if predict_clicked:
        try:
            # Calculate today's sentiment
            today_sentiment = calculate_sentiment_scores(df)
            st.write("📊 Today's News Sentiment Analysis")
            
            sentiment_color = "green" if today_sentiment > 0 else "red"
            st.markdown(f"**Average sentiment score:** ::{sentiment_color}[{today_sentiment:.3f}]")
            
            # Connect to BigQuery and get historical data
            client = get_bq_client(project_id.strip())
            historical_data = fetch_historical_data(client)

            # Clean data and prepare for modeling
            df_clean = historical_data.dropna()
            
            # Train model
            X = df_clean[["aggregated_daily_compound_avg_lag"]]
            y = df_clean["return"]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            
            # Make prediction for tomorrow
            prediction = model.predict([1, today_sentiment])[0]
            
            # Store results in session state
            st.session_state["last_prediction"] = {
                "sentiment": today_sentiment,
                "prediction": prediction,
                "model": model
            }
            
            # Display results
            st.write("🔮 Tomorrow's Prediction")
            pred_color = "green" if prediction > 0 else "red"
            st.markdown(f"**Predicted return:** ::{pred_color}[{prediction:.2%}]")
            
            # Show model details in expander
            with st.expander("View Model Details"):
                st.text(model.summary())

        except Exception as e:
            st.session_state["last_save"] = {"error": f"{type(e).__name__}: {e}"}
            st.error(
                "❌ BigQuery operation failed.\n\n"
                f"{type(e).__name__}: {e}\n\n"
                "Check project/location, IAM, and Secrets formatting."
            )

# Persist last result so it doesn’t disappear on rerun
if st.session_state["last_save"]:
    info = st.session_state["last_save"]
    if "error" in info:
        with st.expander("Last BigQuery error"):
            st.code(info["error"])
    else:
        with st.expander("Last BigQuery write"):
            st.write(f"Target: `{info['target']}`")
            st.write(f"Attempted: {info['attempted']} • New rows: {info['new_rows']}")
            st.markdown(f"[Open in BigQuery]({info['console_url']})")


