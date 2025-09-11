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
import plotly.graph_objects as go
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

st.set_page_config(page_title="S&P 500 News & Predictions", page_icon="📈", layout="wide")

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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_historical_data(_client: bigquery.Client) -> pd.DataFrame:
    """Fetch historical price and sentiment data from BigQuery.
    Note: Leading underscore in _client prevents Streamlit from hashing the client object.
    """
    query = """
    SELECT 
        Date,
        daily_return_pct as return,
        aggregated_daily_compound_avg,
        LAG(aggregated_daily_compound_avg, 1) OVER(ORDER BY Date) as aggregated_daily_compound_avg_lag
    FROM `project-market-news.Final_data_do_not_touch.final_merged_table`
    WHERE Date IS NOT NULL
    ORDER BY Date
    """
    df = _client.query(query).to_dataframe()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

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

# Help tooltip in sidebar
with st.sidebar:
    st.info("""
    ℹ️ **How it works:**
    1. Fetches latest S&P 500 news
    2. Analyzes sentiment using VADER
    3. Uses historical sentiment-return relationship to predict tomorrow's returns
    """)

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
    
    # Add tooltips for technical terms
    st.info("""
    📊 **Technical Terms:**
    - **Sentiment Score:** Using VADER's compound score (numerical value from -1 to +1), not just positive/negative classification
    - **Statistical Significance:** p-value < 0.05 indicates reliable prediction
    - **R-squared:** Shows how well sentiment explains return variation
    """)
    
    predict_clicked = st.button("Analyze Sentiment & Predict Returns", type="primary")

    if predict_clicked:
        if not project_id:
            st.error("⚠️ Please enter your GCP Project ID in the sidebar first.")
            st.stop()
            
        try:
            progress_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            # Step 1: Calculate sentiment
            progress_placeholder.text("Step 1/4: Calculating sentiment scores...")
            progress_bar.progress(25)
            titles = df['title'].tolist()
            analyzer = SentimentIntensityAnalyzer()
            compound_scores = [analyzer.polarity_scores(text)["compound"] for text in titles]
            compound_avg = sum(compound_scores) / len(compound_scores)
            
            # Step 2: Calculate aggregated sentiment
            progress_placeholder.text("Step 2/4: Calculating aggregated sentiment...")
            progress_bar.progress(50)
            today_sentiment = compound_avg
            
            # Step 3: Fetch historical data
            progress_placeholder.text("Step 3/4: Fetching historical data...")
            progress_bar.progress(75)
            client = get_bq_client(project_id.strip())
            historical_data = fetch_historical_data(client)
            df_clean = historical_data.dropna()
            
            if len(df_clean) < 30:  # Minimum sample size check
                progress_placeholder.empty()
                progress_bar.empty()
                st.warning("⚠️ Not enough historical data for reliable prediction (need at least 30 days).")
                st.stop()
            
            # Step 4: Train model and predict
            progress_placeholder.text("Step 4/4: Training model and generating prediction...")
            progress_bar.progress(100)
            X = df_clean[["aggregated_daily_compound_avg_lag"]]
            y = df_clean["return"]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            
            # Model validation
            r2 = model.rsquared
            f_pvalue = model.f_pvalue
            
            if f_pvalue > 0.05:
                st.warning("⚠️ Model may not be statistically significant (p > 0.05)")
                st.info("📌 This means the relationship between sentiment and returns might be weak.")
            
            # Make prediction
            try:
                prediction = model.predict([1, today_sentiment])[0]
            except Exception as e:
                progress_placeholder.empty()
                progress_bar.empty()
                st.error("Failed to make prediction. Error: " + str(e))
                st.stop()
                
            # Store results and clear progress indicators
            progress_placeholder.empty()
            progress_bar.empty()
            
            st.session_state["last_prediction"] = {
                "sentiment": today_sentiment,
                "prediction": prediction,
                "model": model,
                "timestamp": datetime.now()
            }
                
            # Model details (monospace, scrollable, aligned)
            summary = model.summary().as_text()  # or: str(model.summary())
            with st.expander("View Model Details", expanded=True):
                st.code(summary, language="text")  # preserves spacing & adds horiz. scroll

            
            st.subheader("📊 Visual Insights")

            # 1. Gauge Chart for Today's Sentiment
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=today_sentiment,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "#000080"},  # dark blue
                    'steps': [
                        {'range': [-1, -0.5], 'color': "#ff0000"},  # red
                        {'range': [-0.5, 0], 'color': "#ff9999"},  # light red
                        {'range': [0, 0.5], 'color': "#90EE90"},  # light green
                        {'range': [0.5, 1], 'color': "#008000"}   # green
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': today_sentiment
                    }
                },
                title={'text': "Today's Market Sentiment"}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # 2. Predicted Return Gauge
            # Calculate the typical range of returns (e.g., 90% of historical returns)
            returns_percentiles = np.percentile(df_clean['return'], [5, 95])
            min_typical_return = returns_percentiles[0]
            max_typical_return = returns_percentiles[1]
            
            fig_pred = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction * 100,  # Convert to percentage
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {
                        'range': [min_typical_return * 100, max_typical_return * 100],
                        'tickformat': '.1f',
                        'ticksuffix': '%'
                    },
                    'bar': {'color': "#008000" if prediction > 0 else "#ff0000"},
                    'steps': [
                        {'range': [min_typical_return * 100, 0], 'color': "#ff9999"},  # light red
                        {'range': [0, max_typical_return * 100], 'color': "#90EE90"}   # light green
                    ],
                    'threshold': {
                        'line': {'color': "#008000" if prediction > 0 else "#ff0000", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction * 100
                    }
                },
                number={'suffix': '%', 'valueformat': '.2f'},
                title={'text': "Predicted Return (Typical Range: 90% of Historical Returns)"}
            ))
            st.plotly_chart(fig_pred, use_container_width=True)

            # 3. Simpler summary chart for non-technical audience (probability of up day by sentiment bucket)
            q, bin_edges = pd.qcut(
                df_clean['aggregated_daily_compound_avg_lag'], 5, retbins=True, duplicates='drop'
            )
            df_bins = df_clean.copy()
            df_bins['bucket'] = q                           # categorical Interval; preserves order
            df_bins['up'] = (df_bins['return'] > 0).astype(int)

            agg = (
                df_bins.groupby('bucket', observed=True)
                    .agg(up_rate=('up', 'mean'), n=('return', 'size'))
                    .reset_index()
            )
            agg['up_rate_pct'] = (agg['up_rate'] * 100).round(1)

            # find today's bucket and highlight it
            idx = np.searchsorted(bin_edges, today_sentiment, side='right') - 1
            idx = min(max(idx, 0), len(bin_edges) - 2)
            today_iv = pd.Interval(left=bin_edges[idx], right=bin_edges[idx+1], closed='right')
            agg['is_today'] = (agg['bucket'] == today_iv)

            fig_bucket = px.bar(
                agg, x='bucket', y='up_rate_pct', color='is_today',
                color_discrete_map={True: 'green', False: 'lightgray'},
                text='up_rate_pct',
                labels={'bucket': 'Prev-Day Sentiment (quintiles)', 'up_rate_pct': 'Chance market goes up (%)'},
                title='How Often the Market Goes Up Tomorrow by Sentiment Bucket'
            )
            fig_bucket.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
            fig_bucket.update_layout(showlegend=False, yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_bucket, use_container_width=True)


            # Add some explanatory text
            st.markdown("""
            ### 📈 Understanding the Charts
            
            1. **Sentiment Gauge**: Shows today's market sentiment from very negative (-1) to very positive (+1)
            2. **Return Prediction**: Shows predicted return within typical market movement range (90% of historical returns)
            3. **Historical Relationship**: Shows how sentiment relates to returns:
                - Blue dots: Past market returns
                - Red line: Overall trend found by the model
                - Green star: Where today's prediction falls
                - R² value shows how well sentiment explains return variations (higher is better)
            """)

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
