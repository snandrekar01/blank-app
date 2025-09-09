# streamlit_app.py
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote
from datetime import datetime
import hashlib

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
        # prefer news links; skip irrelevant internal anchors
        if "yahoo.com" in href and "/news" not in href:
            continue
        if title:
            items.append({"title": title, "url": href})

    # Deduplicate by URL
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
    """
    Builds a BigQuery client, preferring service-account creds in st.secrets.
    Falls back to ADC if available.
    """
    if "gcp_service_account" in st.secrets:
        sa_info = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project = project_id_input or sa_info.get("project_id")
        return bigquery.Client(project=project, credentials=creds)
    # ADC (e.g., on GCE/Cloud Run); project may be None to use default
    return bigquery.Client(project=project_id_input or None)

def ensure_dataset_and_table(client: bigquery.Client, dataset_id: str, table_id: str, location: str = "EU") -> str:
    """
    Creates dataset/table if not present. Table is partitioned by fetched_at (DAY).
    Returns fully-qualified table id.
    """
    dataset_ref = bigquery.Dataset(f"{client.project}.{dataset_id}")
    dataset_ref.location = location or "EU"
    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        client.create_dataset(dataset_ref)

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
        table.time_partitioning = bigquery.TimePartitioning(
            field="fetched_at", type_=bigquery.TimePartitioningType.DAY
        )
        client.create_table(table)
    return full_table_id

def save_to_bigquery(client: bigquery.Client, df: pd.DataFrame, dataset_id: str, table_id: str) -> dict:
    """
    Loads df into a staging table then MERGEs into target on url_id to avoid duplicates.
    Returns simple stats (attempted, new_rows).
    """
    if df.empty:
        return {"attempted": 0, "new_rows": 0}

    # Prepare dataframe for BQ
    df2 = df.copy()
    df2["url_id"] = df2["url"].apply(lambda u: hashlib.md5(u.encode("utf-8")).hexdigest())
    # BigQuery TIMESTAMP from ISO string
    df2["fetched_at"] = pd.to_datetime(df2["fetched_at"], utc=True)
    df2["added_at"] = pd.Timestamp.utcnow(tz="UTC")

    full_target = f"{client.project}.{dataset_id}.{table_id}"
    staging = f"{full_target}__staging"

    # Load into staging (truncate)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    load_job = client.load_table_from_dataframe(df2, staging, job_config=job_config)
    load_job.result()

    # Count rows that would be inserted (for feedback)
    pre_sql = f"""
      SELECT COUNT(1) AS to_insert
      FROM `{staging}` S
      LEFT JOIN `{full_target}` T
        ON T.url_id = S.url_id
      WHERE T.url_id IS NULL
    """
    pre_res = list(client.query(pre_sql).result())
    to_insert = int(pre_res[0]["to_insert"]) if pre_res else 0

    # Ensure target exists (schema/partitioning)
    ensure_dataset_and_table(client, dataset_id, table_id)

    # MERGE: insert only new URLs
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

    return {"attempted": int(len(df2)), "new_rows": to_insert}

def main():
    st.title("📰 Yahoo Finance News Collector")
    st.caption("Fetch Yahoo Finance headlines and optionally append them to BigQuery (deduped by URL).")

    # Inputs
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL, help="Yahoo Finance ticker (e.g., ^GSPC, AAPL, MSFT)")
    with col2:
        st.write("")
        fetch = st.button("Fetch news", type="primary")

    # BigQuery config (sidebar)
    st.sidebar.header("BigQuery (optional)")
    save_bq = st.sidebar.checkbox("Enable BigQuery save", value=True)
    project_id = st.sidebar.text_input("GCP Project ID", value=st.secrets.get("gcp_project_id", ""))
    dataset_id = st.sidebar.text_input("Dataset", value="market_news")
    table_id = st.sidebar.text_input("Table", value="yahoo_headlines")
    location = st.sidebar.selectbox("Location", ["EU", "US"], index=0, help="Dataset location")

    if fetch:
        if not symbol.strip():
            st.warning("Please enter a symbol.")
            return
        with st.spinner("Fetching headlines…"):
            try:
                df = fetch_yahoo_news(symbol.strip())
            except Exception as e:
                st.error(f"Failed to fetch news: {e}")
                return

        if df.empty:
            st.info("No headlines found. Try another symbol.")
            return

        st.success(f"Fetched {len(df)} headlines for {symbol.strip()}.")
        st.subheader("Headlines")
        for _, row in df.iterrows():
            st.markdown(f"- [{row['title']}]({row['url']})")

        st.subheader("Data")
        st.dataframe(df, use_container_width=True)

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{symbol.strip().replace('^','_')}_yahoo_news_{datetime.utcnow().date()}.csv",
            mime="text/csv",
        )

        # BigQuery save
        if save_bq:
            st.subheader("BigQuery")
            if not dataset_id or not table_id:
                st.warning("Please fill in Dataset and Table in the sidebar.")
            else:
                if st.button("Append to BigQuery"):
                    try:
                        client = get_bq_client(project_id.strip() or None)
                        ensure_dataset_and_table(client, dataset_id.strip(), table_id.strip(), location)
                        stats = save_to_bigquery(client, df, dataset_id.strip(), table_id.strip())
                        st.success(
                            f"Attempted: {stats['attempted']}, newly inserted: {stats['new_rows']} "
                            f"into `{client.project}.{dataset_id}.{table_id}`."
                        )
                    except Exception as e:
                        st.error(f"BigQuery save failed: {e}")

    with st.expander("Notes"):
        st.write(
            "- Store service account JSON under `st.secrets['gcp_service_account']` or use ADC. "
            "A sample secrets format is shown below.\n"
            "- Table is partitioned daily on `fetched_at`; duplicates prevented via a MERGE on `url_id`."
        )
        st.code(
            """
# .streamlit/secrets.toml (example)
gcp_project_id = "your-project-id"

[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
client_email = "svc-acct@your-project-id.iam.gserviceaccount.com"
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"
""",
            language="toml",
        )

if __name__ == "__main__":
    main()
