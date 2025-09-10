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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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

st.set_page_config(page_title="S&P 500 News & Predictions", page_icon="📈", layout="centered")

# ---- helper functions ----------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_yahoo_news(symbol: str) -> pd.DataFrame:
    """Fetch news headlines from Yahoo Finance."""
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

def get_bq_client(project_id_input: str | None) -> bigquery.Client:
    """Initialize BigQuery client."""
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

# ---- app --------------------------------------------------------------------

# Initialize session state
if "news_df" not in st.session_state:
    st.session_state["news_df"] = None
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

st.title("📈 S&P 500 News & Predictions")
st.caption("Analyze S&P 500 news sentiment and predict tomorrow's returns")

# Fetch news button
fetch_clicked = st.button("Fetch Latest S&P 500 News", type="primary")

# Sidebar config
st.sidebar.header("BigQuery Configuration")
project_id = st.sidebar.text_input("GCP Project ID", value=st.secrets.get("gcp_project_id", ""))

# On fetch, get news and store in session
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

# Show data if available
df = st.session_state["news_df"]
if isinstance(df, pd.DataFrame) and not df.empty:
    st.subheader("Today's Headlines")
    for _, row in df.iterrows():
        st.markdown(f"- [{row['title']}]({row['url']})")

    with st.expander("View Raw Data"):
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"SP500_news_{datetime.utcnow().date()}.csv",
            mime="text/csv",
        )

    # Prediction section
    st.subheader("Sentiment Analysis & Predictions")
    predict_clicked = st.button("Analyze Sentiment & Predict Returns", type="primary")

    if predict_clicked:
        try:
            # Calculate today's sentiment
            today_sentiment = calculate_sentiment_scores(df)
            
            st.write("📊 Today's News Sentiment")
            sentiment_color = "green" if today_sentiment > 0 else "red"
            st.markdown(f"**Average sentiment score:** ::{sentiment_color}[{today_sentiment:.3f}]")
            
            with st.spinner("Fetching historical data and training model..."):
                # Get historical data and train model
                client = get_bq_client(project_id.strip())
                historical_data = fetch_historical_data(client)
                df_clean = historical_data.dropna()
                
                # Train model
                X = df_clean[["aggregated_daily_compound_avg_lag"]]
                y = df_clean["return"]
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                
                # Predict
                prediction = model.predict([1, today_sentiment])[0]
                
                # Store results
                st.session_state["last_prediction"] = {
                    "sentiment": today_sentiment,
                    "prediction": prediction,
                    "model": model,
                    "timestamp": datetime.now()
                }
                
                # Show prediction
                st.write("🔮 Tomorrow's Return Prediction")
                pred_color = "green" if prediction > 0 else "red"
                st.markdown(f"**Predicted return:** ::{pred_color}[{prediction:.2%}]")
                
                # Model details
                with st.expander("View Model Details"):
                    st.text(model.summary())
                    
                # Scatter plot
                fig = px.scatter(
                    df_clean,
                    x="aggregated_daily_compound_avg_lag",
                    y="return",
                    title="Historical Sentiment vs Returns",
                    labels={
                        "aggregated_daily_compound_avg_lag": "Previous Day Sentiment",
                        "return": "Return"
                    },
                    trendline="ols"
                )
                st.plotly_chart(fig)

        except Exception as e:
            st.session_state["last_prediction"] = {"error": str(e)}
            st.error(
                "❌ Prediction failed.\n\n"
                f"Error: {str(e)}\n\n"
                "Please check your BigQuery connection and permissions."
            )

# Show last prediction if available
elif "last_prediction" in st.session_state and st.session_state["last_prediction"]:
    info = st.session_state["last_prediction"]
    if "error" in info:
        with st.expander("Last Error"):
            st.code(info["error"])
    else:
        with st.expander("Last Prediction"):
            st.write(f"Analysis from: {info['timestamp']}")
            sentiment_color = "green" if info["sentiment"] > 0 else "red"
            pred_color = "green" if info["prediction"] > 0 else "red"
            st.markdown(f"**Sentiment score:** ::{sentiment_color}[{info['sentiment']:.3f}]")
            st.markdown(f"**Predicted return:** ::{pred_color}[{info['prediction']:.2%}]")
