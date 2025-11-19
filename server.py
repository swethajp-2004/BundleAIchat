# server.py -- MotherDuck / DuckDB updated


import os
import re
import json
import traceback
from html import escape
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, jsonify, send_from_directory, g
import duckdb
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import calendar
import time
from utils import rows_to_html_table, plot_to_base64, embed_text
import uuid
from pathlib import Path
import sqlite3

# debug flag for safe SQL visibility
DEBUG_SQL = os.getenv("DEBUG_SQL", "false").lower() in ("1", "true", "yes")

# -------------------------
# Basic setup
# -------------------------
MEMORY_FILE = Path('./data/conversations.json')
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
print("PWD:", os.getcwd())

TZ = ZoneInfo("Asia/Kolkata")  # user timezone

# -------------------------
# Helpers: dataset anchor date and month/year parsing
# -------------------------

def get_dataset_anchor_datetime(date_col_name: str, table_name: str, conn: duckdb.DuckDBPyConnection):
    """Return a timezone-aware datetime anchored to the dataset's latest date (MAX(date_col)).
    If the dataset doesn't expose a date, fall back to current time in TZ.
    """
    try:
        if not date_col_name:
            raise ValueError("No date column specified")
        q = f"SELECT MAX(CAST({date_col_name} AS DATE)) FROM {table_name};"
        r = conn.execute(q).fetchone()
        if r and r[0]:
            d = r[0]
            # convert to datetime at start of day in TZ
            return datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=TZ)
    except Exception:
        # ignore and fallback
        pass
    return datetime.now(TZ)

# month/year parsing utilities
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
ABBR_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}

def parse_month_year_from_text(text: str):
    """Return (month_int, year_int) where either can be None.
    Examples:
      'july 2025' -> (7, 2025)
      'july' -> (7, None)
      None detected -> (None, None)
    """
    text_l = (text or "").lower()
    m = None
    # try full names first
    for name, idx in MONTHS.items():
        if re.search(rf"\b{name}\b", text_l):
            m = idx
            break
    if not m:
        for name, idx in ABBR_MONTHS.items():
            if re.search(rf"\b{name}\b", text_l):
                m = idx
                break
    y = None
    y_m = re.search(r"\b(20\d{2})\b", text or "")
    if y_m:
        y = int(y_m.group(1))
    return m, y

# -------------------------
# Relative-time helpers (minimal changes)
# parse_relative_time_phrase now accepts an anchor datetime
# -------------------------

def _start_of_month(dt: datetime) -> date:
    return date(dt.year, dt.month, 1)

def _end_of_month(dt: datetime) -> date:
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    return date(dt.year, dt.month, last_day)

def month_n_ago(today_dt: datetime, n: int):
    y = today_dt.year
    m = today_dt.month - n
    while m <= 0:
        m += 12
        y -= 1
    return y, m

def parse_relative_time_phrase(question: str, anchor_dt: datetime = None):
    """Parse relative time phrases but anchor them to anchor_dt when provided.
    Returns a dict describing the type and meta. If a month name is provided without a year
    the returned dict will contain 'month_name' and year=None so caller can ask the user for the year.
    """
    q = (question or "").lower()
    now = anchor_dt or datetime.now(TZ)

    if re.search(r"\b(what(?:'s| is)? the time|what time is it|current time|time now|tell me the time|time\?)\b", q):
        return {"type": "now", "datetime": now}
    if re.search(r"\b(today|what(?:'s| is)? the date|current date|date today|todays date|what is the date today)\b", q):
        return {"type": "day", "date": now.date().isoformat(), "datetime": now}

    m = re.search(r'\b(\d+)\s+days?\s+ago\b', q)
    if m:
        n = int(m.group(1))
        target = (now - timedelta(days=n)).date()
        return {"type": "day", "date": target.isoformat()}
    if re.search(r'\byesterday\b', q):
        target = (now - timedelta(days=1)).date()
        return {"type": "day", "date": target.isoformat()}

    m = re.search(r'\b(\d+)\s+months?\s+ago\b', q)
    if m:
        n = int(m.group(1))
        y, mth = month_n_ago(now, n)
        return {"type": "month", "month": f"{y:04d}-{mth:02d}"}
    if re.search(r'\b(previous|last)\s+month\b', q):
        y, mth = month_n_ago(now, 1)
        return {"type": "month", "month": f"{y:04d}-{mth:02d}"}
    m = re.search(r'\b(\d+)(?:st|nd|rd|th)?\s+previous\s+month\b', q)
    if m:
        n = int(m.group(1))
        y, mth = month_n_ago(now, n)
        return {"type": "month", "month": f"{y:04d}-{mth:02d}"}
    if re.search(r"\b(this|current)\s+month\b", q):
        y, mth = month_n_ago(now, 0)
        return {"type": "month", "month": f"{y:04d}-{mth:02d}"}
    m = re.search(r"\blast\s+(\d+)\s+months\b", q)
    if m:
        n = int(m.group(1))
        start_y, start_m = month_n_ago(now, n)
        start = _start_of_month(datetime(start_y, start_m, 1))
        end_y, end_m = month_n_ago(now, 1)
        end = _end_of_month(datetime(end_y, end_m, 1))
        return {"type": "range", "start": start.isoformat(), "end": end.isoformat()}

    # explicit month mention (e.g., 'for july 2025' or 'july 2025' or just 'july')
    # returns month_name_only if year not present
    parsed_month, parsed_year = parse_month_year_from_text(q)
    if parsed_month:
        if parsed_year:
            return {"type": "month", "month": f"{parsed_year:04d}-{parsed_month:02d}"}
        else:
            return {"type": "month_name_only", "month_name": calendar.month_name[parsed_month], "month_num": parsed_month}
    
    return None

def handle_relative_time(question: str, sale_date_col="SaleDate"):
    # compute anchor based on dataset
    anchor = get_dataset_anchor_datetime(sale_date_col, TABLE, con) if 'con' in globals() and TABLE in globals() else datetime.now(TZ)
    parsed = parse_relative_time_phrase(question, anchor_dt=anchor)
    if not parsed:
        return None
    if parsed.get("type") == "now":
        dt = parsed.get("datetime")
        return {"action": "now", "text": dt.strftime("%Y-%m-%d %H:%M:%S %Z"), "meta": parsed}
    if parsed.get("type") == "day":
        d = parsed.get("date")
        cond = f"CAST({sale_date_col} AS DATE) = DATE '{d}'"
        return {"action": "filter", "filter_sql": cond, "meta": parsed}
    if parsed.get("type") == "month":
        mon = parsed.get("month")  # 'YYYY-MM'
        cond = f"STRFTIME('%Y-%m',{sale_date_col}) = '{mon}'"
        return {"action": "filter", "filter_sql": cond, "meta": parsed}
    if parsed.get("type") == "month_name_only":
        # user said 'July' with no year — do not guess. Ask user to specify year.
        mn = parsed.get("month_name")
        return {"action": "ask_year", "message": f"Please specify which year you mean for '{mn}' (for example: 'July 2025')."}
    if parsed.get("type") == "range":
        s = parsed.get("start")
        e = parsed.get("end")
        cond = f"CAST({sale_date_col} AS DATE) BETWEEN DATE '{s}' AND DATE '{e}'"
        return {"action": "filter", "filter_sql": cond, "meta": parsed}
    return None

# -------------------------
# Load environment, DB, OpenAI client
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DUCKDB_FILE = os.getenv("DUCKDB_FILE", "./data/sales.duckdb")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "3000"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Connect DuckDB (support md: URIs and local files)
# -------------------------
db_path = DUCKDB_FILE.strip()
if not (db_path.lower().startswith("md:") or db_path.lower().startswith("md://")):
    db_path = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

con = duckdb.connect(db_path)

# -------------------------
# Detect available tables and set TABLE to sales_dataset
# -------------------------
available = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
print("Detected available tables:", available)
TABLE = "sales_dataset"
if TABLE not in available:
    # If exact name not present, try case-insensitive match
    for t in available:
        if t.lower() == TABLE.lower():
            TABLE = t
            break

# -------------------------
# Schema & sample
# -------------------------
cols_info = con.execute(f"PRAGMA table_info('{TABLE}')").fetchall()
COLS = [c[1] for c in cols_info] if cols_info else []
SAMPLE_DF = con.execute(f"SELECT * FROM {TABLE} LIMIT 10").fetchdf() if COLS else pd.DataFrame()
try:
    total_count = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
except Exception:
    total_count = None
print(f"Using table: {TABLE} rows={total_count}, cols={len(COLS)}")

# -------------------------
# Auto-detect common columns
# -------------------------
def find_best(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand and cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def get_profit_col():
    """Return the best profit-like column name or None."""
    # First prefer explicitly-detected column if present in actual columns
    p = DETECTED.get('profit')
    if p and p in COLS:
        return p
    # fallback to common names
    for candidate in ['Profit', 'total_profit', 'profit_amount', 'ProfitAmount']:
        if candidate in COLS:
            return candidate
    return None

# Candidate lists
REVENUE_CANDS = ['Revenue', 'ProviderPaid', 'NetAfterCb', 'GrossAfterCb', 'Amount', 'SaleAmount']
PROFIT_CANDS = ['Profit', 'NetProfit', 'ProfitAmount', 'Margin']
DATE_CANDS = ['sale_date_parsed', 'SaleDate', 'Date', 'OrderDate']
CHANNEL_CANDS = ['MainChannel', 'SalesChannel', 'Channel', 'ChannelName']
INSTALL_CANDS = ['install_date_parsed', 'InstallDate', 'IsInstalled']
MARKETING_CANDS = [
    'Campaign', 'CampaignName', 'UtmCampaign', 'MarketingCampaign',
    'Marketing_Source', 'SourceCampaign', 'Campaign_Name', 'MarketingCampaignName',
    'marketing campaign', 'marketing', 'utm_campaign', 'utm_campaign_name', 'marketting'
]
PACKAGE_CANDS = ['Package', 'Product', 'Plan', 'Offering']

# Explicit marketing source / portal campaign candidates
MARKETING_SOURCE_CANDS = [
    'MarketingSource', 'Marketing Source', 'Marketing_Source'
]
PORTAL_CAMPAIGN_CANDS = [
    'PortalCampaignName', 'Portal Campaign Name', 'Portal_Campaign_Name',
    'PortalCampaign', 'Portal Campaign'
]

# Build DETECTED map using the candidate lists
DETECTED = {
    'date': find_best(COLS, DATE_CANDS),
    'revenue': find_best(COLS, REVENUE_CANDS),
    'profit': find_best(COLS, PROFIT_CANDS),
    'channel': find_best(COLS, CHANNEL_CANDS),
    'campaign': find_best(COLS, MARKETING_CANDS),
    'package': find_best(COLS, PACKAGE_CANDS),
    'install': find_best(COLS, INSTALL_CANDS),
    'marketing_source': find_best(COLS, MARKETING_SOURCE_CANDS),
    'portal_campaign': find_best(COLS, PORTAL_CAMPAIGN_CANDS),
}

print("Auto-detected columns:", DETECTED)

# ------------------------------------------------------------------
# NEW: helper to clean extracted entity values (packages, sources, etc.)
# ------------------------------------------------------------------
def _clean_entity_value(value: str) -> str | None:
    """
    Clean up extracted entity values by:
      - trimming whitespace and quotes
      - removing trailing time phrases like 'for 2024', 'for last month', etc.
      - collapsing multiple spaces
    """
    if not value:
        return None
    v = value.strip().strip('"').strip("'")
    # Cut off common trailing phrases such as "for 2024", "in July", "for last month"...
    v = re.split(r'\b(for|in|during|on|at|from|by)\b', v, 1)[0].strip()
    # collapse multiple spaces
    v = re.sub(r'\s+', ' ', v)
    return v or None

def _extract_entity_value(question: str, entity_type: str) -> str | None:
    """
    Try to extract the entity value (package name, marketing source, portal campaign, channel, etc.)
    from flexible natural language like:
      - 'show me the details about fast package'
      - 'show me the details for the package fast'
      - 'marketing source Google for 2024'
      - 'main channel retail for last month'
    entity_type: 'package', 'marketing_source', 'portal_campaign', 'channel'
    """
    text = question or ""

    # ----- PACKAGE -----
    if entity_type == "package":
        # e.g. "for the package fast", "the package fast"
        m = re.search(r'\b(?:for\s+the\s+|the\s+)?package\b\s+([A-Za-z0-9 _\-\(\)/]+)', text, flags=re.IGNORECASE)
        if m:
            return _clean_entity_value(m.group(1))
        # e.g. "details about fast package"
        m = re.search(r'\babout\s+([A-Za-z0-9 _\-\(\)/]+?)\s+\bpackage\b', text, flags=re.IGNORECASE)
        if m:
            return _clean_entity_value(m.group(1))

    # ----- MARKETING SOURCE -----
    elif entity_type == "marketing_source":
        # 'marketing source <name>'
        m = re.search(r'\bmarketing\s+source\b\s+([A-Za-z0-9 _\-\(\)/]+)', text, flags=re.IGNORECASE)
        if m:
            return _clean_entity_value(m.group(1))
        # '<name> marketing source'
        m = re.search(r'([A-Za-z0-9 _\-\(\)/]+)\s+\bmarketing\s+source\b', text, flags=re.IGNORECASE)
        if m:
            return _clean_entity_value(m.group(1))

    # ----- PORTAL CAMPAIGN -----
    elif entity_type == "portal_campaign":
        # 'portal campaign name <name>' or 'portal campaign <name>'
        m = re.search(r'\bportal\s+campaign(?:\s+name)?\b\s+([A-Za-z0-9 _\-\(\)/]+)', text, flags=re.IGNORECASE)
        if m:
            return _clean_entity_value(m.group(1))
        # '<name> portal campaign'
        m = re.search(r'([A-Za-z0-9 _\-\(\)/]+)\s+\bportal\s+campaign(?:\s+name)?\b', text, flags=re.IGNORECASE)
        if m:
            return _clean_entity_value(m.group(1))

    # ----- CHANNEL / MAIN CHANNEL -----
    elif entity_type == "channel":
        # 'main channel retail ...'
        m = re.search(r'\bmain\s+channel\b\s+([A-Za-z0-9 _\-\(\)/]+)', text, flags=re.IGNORECASE)
        if m:
            return _clean_entity_value(m.group(1))
        # 'channel retail ...'
        m = re.search(r'\bchannel\b\s+([A-Za-z0-9 _\-\(\)/]+)', text, flags=re.IGNORECASE)
        if m:
            return _clean_entity_value(m.group(1))
        # '<name> channel'
        m = re.search(r'([A-Za-z0-9 _\-\(\)/]+)\s+\bchannel\b', text, flags=re.IGNORECASE)
        if m:
            return _clean_entity_value(m.group(1))

    # as an extra safety, try quotes: "... fast ..." etc.
    m = re.search(r'"([^"]+)"', text)
    if m:
        return _clean_entity_value(m.group(1))

    return None

def choose_marketing_column(question, prefer=None):
    """
    Decide which marketing-related column to use based on the question text.
    We prefer:
      - marketing_source column when they say "marketing source"
      - portal_campaign column when they say "portal campaign"
      - otherwise fallback to generic DETECTED['campaign'].
    """
    ql = (question or "").lower()
    marketing_col = DETECTED.get('marketing_source')
    portal_col = DETECTED.get('portal_campaign')

    # explicit phrases
    if marketing_col and (prefer == "marketing_source" or re.search(r'\bmarketing source(s)?\b', ql)):
        return marketing_col
    if portal_col and (prefer == "portal_campaign" or re.search(r'\bportal campaign(s)?\b', ql)):
        return portal_col

    # weaker hints
    if marketing_col and re.search(r'\bsource\b', ql):
        return marketing_col
    if portal_col and re.search(r'\bportal\b', ql):
        return portal_col

    # fallback
    return marketing_col or portal_col or DETECTED.get('campaign')

def run_sql_and_fetch_df(sql: str):
    print("Executing SQL → (hidden unless DEBUG_SQL is true)")
    try:
        df = con.execute(sql).fetchdf()
        return df
    except Exception as e:
        raise

# -------------------------
# Safety & small helpers
# -------------------------
def handle_entity_details(question):
    """
    Returns details for one package, one marketing source, one portal campaign,
    or one channel (main channel).
    Example:
        "show me the details about fast package"
        "show me the performance of marketing source Google for last month"
        "summary of portal campaign name ABC in 2024"
        "give me the details for the main channel retail for last month"
    """

    ql = (question or "").lower()
    date_col = DETECTED.get('date') or 'SaleDate'

    # 1) Detect which entity type the user is talking about
    if "marketing source" in ql:
        entity_type = "marketing_source"
        col = DETECTED.get("marketing_source")
    elif "portal campaign" in ql or "portal campaign name" in ql:
        entity_type = "portal_campaign"
        col = DETECTED.get("portal_campaign")
    elif "main channel" in ql or re.search(r'\bchannel\b', ql):
        entity_type = "channel"
        col = DETECTED.get("channel")
    elif "package" in ql:
        entity_type = "package"
        col = DETECTED.get("package")
    else:
        return {
            "reply": "Please specify whether you want details for a package, marketing source, main channel, or portal campaign.",
            "table_html": "",
            "plot_data_uri": None
        }

    if not col:
        return {"reply": f"I could not find a column for '{entity_type}' in your dataset.", "table_html": "", "plot_data_uri": None}

    # 2) Extract the exact value, e.g., "fast", "Google", "ABC", "retail"
    target_value = _extract_entity_value(question, entity_type)
    if not target_value:
        return {
            "reply": f"I couldn't figure out which {entity_type.replace('_', ' ')} you mean. Example: 'package fast', 'main channel retail', or 'marketing source Google'.",
            "table_html": "",
            "plot_data_uri": None
        }

    # 3) Apply time filter (month/year/relative)
    month, year = parse_month_year_from_text(question)
    rel = handle_relative_time(question, sale_date_col=date_col)

    if month and not year:
        return {
            "reply": f"Please specify the year for '{calendar.month_name[month]}'. Example: 'July 2025'.",
            "table_html": "",
            "plot_data_uri": None
        }

    if month and year:
        last_day = calendar.monthrange(year, month)[1]
        start = date(year, month, 1).isoformat()
        end = date(year, month, last_day).isoformat()
        date_filter = f"CAST({date_col} AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'"
        period_text = f" for {calendar.month_name[month]} {year}"

    elif year:
        date_filter = f"STRFTIME('%Y',{date_col}) = '{year}'"
        period_text = f" for {year}"

    elif rel and rel.get("action") == "filter":
        date_filter = rel["filter_sql"]
        period_text = " for the selected period"

    else:
        # No date mentioned → use complete dataset
        date_filter = "1=1"
        period_text = ""

    # 4) Final SQL
    install_condition = "(install_date_parsed IS NOT NULL OR InstallDate IS NOT NULL OR IsInstalled = 1)"
    profit_col = get_profit_col()
    revenue_col = DETECTED.get("revenue") or "NetAfterCb"

    # case-insensitive match for entity value
    safe_target = target_value.replace("'", "''")
    where_entity = f"LOWER(TRIM(COALESCE({col},''))) = LOWER('{safe_target}')"

    sql = f"""
        SELECT
            STRFTIME('%Y-%m',{date_col}) AS month,
            COUNT(*) AS orders,
            SUM(CASE WHEN {install_condition} THEN 1 ELSE 0 END) AS installations,
            SUM(CASE WHEN DisconnectDate IS NOT NULL THEN 1 ELSE 0 END) AS disconnections,
            SUM(CAST({revenue_col} AS DOUBLE)) AS revenue,
            SUM(CAST({profit_col} AS DOUBLE)) AS profit
        FROM {TABLE}
        WHERE {where_entity}
        AND {date_filter}
        GROUP BY month
        ORDER BY month;
    """

    try:
        df = run_sql_and_fetch_df(sql)
    except Exception as e:
        _log_and_mask_error(e, "entity_details_sql")
        return _safe_user_error("I couldn't fetch details for that entity.")

    if df is None or df.empty:
        return {
            "reply": f"No data found for {entity_type.replace('_',' ')} '{target_value}'{period_text}.",
            "table_html": "",
            "plot_data_uri": None
        }

    # summary row
    total_orders = int(df["orders"].sum())
    total_inst = int(df["installations"].sum())
    total_disc = int(df["disconnections"].sum())
    total_rev = float(df["revenue"].sum())
    total_profit = float(df["profit"].sum())

    summary = (
        f"For {entity_type.replace('_', ' ')} '{target_value}'{period_text}: "
        f"total orders = {total_orders:,}, installations = {total_inst:,}, "
        f"disconnections = {total_disc:,}, revenue ≈ {total_rev:,.0f}, "
        f"profit ≈ {total_profit:,.0f}"
    )

    table_html = rows_to_html_table(df.to_dict(orient="records"))

    # Plot
    plot_uri = plot_to_base64(
        df["month"].astype(str).tolist(),
        df["revenue"].fillna(0).tolist(),
        kind="line",
        title=f"Performance of {target_value}"
    )

    resp = {
        "reply": summary,
        "table_html": table_html,
        "plot_data_uri": plot_uri,
        "rows_returned": len(df)
    }
    return resp

def _safe_user_error(msg=None):
    return {"reply": msg or "Sorry — I couldn't get that data. Please try rephrasing or ask for fewer filters.", "table_html": "", "plot_data_uri": None}

def _log_and_mask_error(e, context=""):
    print("INTERNAL ERROR:", context, str(e), traceback.format_exc())
    return "I couldn't complete that query due to an internal error. Try rephrasing or ask a simpler question."

def _year_from_question(text, default_offset=1):
    m = re.search(r"\b(20\d{2})\b", (text or ""))
    if m:
        return int(m.group(1))
    # no explicit year — do not silently guess; return None so caller can decide
    return None

def _format_topk_table(df, label_col, value_col):
    rows = df[[label_col, value_col]].fillna("").to_dict(orient='records')
    table_html = rows_to_html_table(rows)
    return table_html, rows

def _package_null_filter(include_null=False, colname="Package"):
    if include_null:
        return "1=1"
    else:
        return f"({colname} IS NOT NULL AND TRIM(COALESCE({colname},'')) <> '')"

def _last_year_guard(text):
    """
    If the question says 'last year' but does NOT contain an explicit 20xx year,
    we should NOT guess. Ask user to specify the year.
    """
    ql = (text or "").lower()
    if re.search(r'\blast year\b', ql) and not re.search(r'\b20\d{2}\b', ql):
        return {
            "reply": "Please specify which year you mean by 'last year' (for example: '2024').",
            "table_html": "",
            "plot_data_uri": None
        }
    return None

def handle_top_packages_by_installations(question):
    try:
        ql = (question or "").lower()

        # parse month/year
        month, year = parse_month_year_from_text(question)

        # parse limit
        m = re.search(r'\btop\s+(\d{1,2})\b', ql)
        limit = int(m.group(1)) if m else 10

        include_null = bool(re.search(r'\b(include null|include nulls|include missing|include null package)\b', ql))
        date_col = DETECTED.get('date') or 'SaleDate'
        null_clause = "" if include_null else f"AND {_package_null_filter(include_null=False, colname='Package')}"

        # ---- STRICT: month WITHOUT year -> ask user ----
        if month and not year:
            month_name = calendar.month_name[month]
            return {
                "reply": f"Please specify the year for '{month_name}' (for example: 'July 2025') so I can return that exact month.",
                "table_html": "",
                "plot_data_uri": None
            }

        # ---- CASE 1: explicit month + year ----
        if month and year:
            last_day = calendar.monthrange(year, month)[1]
            start = date(year, month, 1)
            end = date(year, month, last_day)
            date_filter = f"CAST({date_col} AS DATE) BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'"
            period_text = f" for {calendar.month_name[month]} {year} (from {start.isoformat()} to {end.isoformat()})"

        # ---- CASE 2: explicit year ONLY ----
        elif year and not month:
            date_filter = f"STRFTIME('%Y', {date_col}) = '{year}'"
            period_text = f" in {year}"

        else:
            # ---- CASE 3: try relative phrases like 'last month', 'this month' etc. ----
            rel = handle_relative_time(question, sale_date_col=date_col)
            if rel and rel.get("action") == "filter":
                date_filter = rel['filter_sql']
                period_text = " for the selected period"
            else:
                # ---- CASE 4: no time info at all -> NO DATE FILTER (entire dataset) ----
                date_filter = "1=1"
                period_text = ""

        sql = f"""
            SELECT Package AS package,
                   SUM(CASE WHEN (install_date_parsed IS NOT NULL OR InstallDate IS NOT NULL OR IsInstalled = 1) THEN 1 ELSE 0 END) AS installations
            FROM {TABLE}
            WHERE {date_filter} {null_clause}
            GROUP BY package
            ORDER BY installations DESC
            LIMIT {limit};
        """
        try:
            df = run_sql_and_fetch_df(sql)
        except Exception as e:
            _log_and_mask_error(e, "top_packages_sql_exec")
            return _safe_user_error("I couldn't compute top packages right now. Please try rephrasing or ask for a different period.")

        if df is None or df.empty:
            return {
                "reply": f"No installation data found{period_text or ''}.",
                "table_html": "",
                "plot_data_uri": None
            }

        table_html, rows = _format_topk_table(df, 'package', 'installations')
        top_list = [f"{r['package'] or '<NULL>'}: {int(r['installations']):,}" for r in rows]
        explanation = f"Top {len(rows)} packages by installations{period_text}:\n" + "; ".join(top_list)
        resp = {
            "reply": explanation,
            "table_html": table_html,
            "plot_data_uri": None,
            "rows_returned": len(rows)
        }
        if DEBUG_SQL:
            resp["debug_sql"] = sql
        return resp

    except Exception as e:
        return _safe_user_error(_log_and_mask_error(e, "top_packages_by_installations"))


def handle_top_packages_by_profit(question):
    """
    Top N packages based on profit (year/month filters handled elsewhere).
    """
    try:
        m = re.search(r'\btop\s+(\d{1,2})\b', question.lower())
        limit = int(m.group(1)) if m else 10

        package_col = DETECTED.get('package') or 'Package'
        profit_col = get_profit_col()
        date_col = DETECTED.get('date') or 'SaleDate'
        if not profit_col:
            return {"reply":"No profit-like column detected in the dataset.", "table_html":"", "plot_data_uri": None}

        # Optional time filter: check for explicit month/year in question
        month, year = parse_month_year_from_text(question)
        if month and not year:
            # strict: ask for year
            return {"reply": f"Please specify the year for '{calendar.month_name[month]}'. (e.g., 'July 2025')", "table_html":"", "plot_data_uri": None}

        if month and year:
            last_day = calendar.monthrange(year, month)[1]
            start = date(year, month, 1)
            end = date(year, month, last_day)
            date_filter = f"AND CAST({date_col} AS DATE) BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'"
            period_text = f" for {calendar.month_name[month]} {year} (from {start.isoformat()} to {end.isoformat()})"
        else:
            date_filter = ""
            period_text = ""

        sql = f"""
            SELECT {package_col} AS package,
                   SUM(CAST(COALESCE({profit_col},0) AS DOUBLE)) AS total_profit,
                   COUNT(*) AS orders
            FROM {TABLE}
            WHERE ({package_col} IS NOT NULL AND TRIM(COALESCE({package_col},'')) <> '') {date_filter}
            GROUP BY {package_col}
            ORDER BY total_profit DESC
            LIMIT {limit};
        """
        try:
            df = run_sql_and_fetch_df(sql)
        except Exception as e:
            _log_and_mask_error(e, "top_packages_by_profit_sql_exec")
            return _safe_user_error("I couldn't compute top packages by profit right now. Please try rephrasing.")

        if df is None or df.empty:
            return {"reply": f"No profit data found{period_text}.", "table_html":"", "plot_data_uri": None}

        table_html = rows_to_html_table(df.to_dict(orient='records'))
        rows = df.to_dict(orient='records')
        top_list = [f"{r.get('package') or '<NULL>'}: ${float(r.get('total_profit') or 0):,.0f}" for r in rows]
        explanation = f"Top {len(rows)} packages by profit{period_text}:\n" + "; ".join(top_list)
        resp = {"reply": explanation, "table_html": table_html, "plot_data_uri": None, "rows_returned": len(rows)}
        if DEBUG_SQL:
            resp["debug_sql"] = sql
        return resp
    except Exception as e:
        return _safe_user_error(_log_and_mask_error(e, "handle_top_packages_by_profit"))

def handle_top_marketing_campaigns_by_installation(question):
    """
    Return top N marketing campaigns by installations, honoring month/year or relative time.
    Uses marketing source / portal campaign / generic campaign depending on question.
    """
    try:
        guard = _last_year_guard(question)
        if guard:
            return guard

        # limit
        m = re.search(r'\btop\s+(\d{1,2})\b', question.lower())
        limit = int(m.group(1)) if m else 10

        date_col = DETECTED.get('date') or 'SaleDate'
        campaign_col = choose_marketing_column(question)
        if not campaign_col:
            return {"reply":"I couldn't find a marketing/campaign column in the dataset. Try 'Marketing Source' or 'Portal Campaign Name' in your data.", "table_html":"", "plot_data_uri": None}

        # parse month/year or year-only
        month, year = parse_month_year_from_text(question)
        if not year:
            my = re.search(r'\b(20\d{2})\b', question)
            if my:
                year = int(my.group(1))

        # strict: if month without year -> ask
        if month and not year:
            return {"reply": f"Please specify the year for '{calendar.month_name[month]}' (for example: 'July 2025').", "table_html":"", "plot_data_uri": None}

        # installations condition
        install_condition = "(install_date_parsed IS NOT NULL OR InstallDate IS NOT NULL OR IsInstalled = 1)"

        # build WHERE clause
        where_clauses = [f"({campaign_col} IS NOT NULL AND TRIM(COALESCE({campaign_col},'')) <> '')"]

        period_text = ""
        if month and year:
            last_day = calendar.monthrange(year, month)[1]
            start = date(year, month, 1)
            end = date(year, month, last_day)
            where_clauses.append(f"CAST({date_col} AS DATE) BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'")
            period_text = f" for {calendar.month_name[month]} {year} (from {start.isoformat()} to {end.isoformat()})"
        elif year and not month:
            where_clauses.append(f"STRFTIME('%Y', {date_col}) = '{year}'")
            period_text = f" for {year}"
        else:
            # try relative parsing
            rel = handle_relative_time(question, sale_date_col=date_col)
            if rel and rel.get('action') == 'filter':
                where_clauses.append(rel['filter_sql'])
                period_text = " for the selected period"

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        sql = f"""
            SELECT {campaign_col} AS campaign,
                   SUM(CASE WHEN {install_condition} THEN 1 ELSE 0 END) AS installations
            FROM {TABLE}
            WHERE {where_sql}
            GROUP BY {campaign_col}
            ORDER BY installations DESC
            LIMIT {limit};
        """
        try:
            df = run_sql_and_fetch_df(sql)
        except Exception as e:
            _log_and_mask_error(e, "top_marketing_campaigns_sql_exec")
            return _safe_user_error("I couldn't compute top marketing campaigns right now. Please try rephrasing.")

        if df is None or df.empty:
            return {"reply": f"No installation data found grouped by campaign{period_text}.", "table_html":"", "plot_data_uri": None}

        table_html, rows = _format_topk_table(df, 'campaign', 'installations')
        top_list = [f"{r['campaign'] or '<NULL>'}: {int(r['installations']):,}" for r in rows]
        explanation = f"Top {len(rows)} marketing campaigns by installations{period_text}:\n" + "; ".join(top_list)
        resp = {"reply": explanation, "table_html": table_html, "plot_data_uri": None, "rows_returned": len(rows)}
        if DEBUG_SQL:
            resp["debug_sql"] = sql
        return resp
    except Exception as e:
        return _safe_user_error(_log_and_mask_error(e, "handle_top_marketing_campaigns_by_installation"))

def handle_monthly_revenue_last_year(question):
    try:
        guard = _last_year_guard(question)
        if guard:
            return guard

        # prefer explicit year
        year = _year_from_question(question, default_offset=1)
        if year is None:
            # if user asked 'last year' or similar, interpret using dataset anchor
            rel = handle_relative_time(question, sale_date_col=(DETECTED.get('date') or 'SaleDate'))
            if rel and rel.get('action') == 'filter':
                date_col = DETECTED.get('date') or 'SaleDate'
                rev_col = DETECTED.get('revenue') if DETECTED.get('revenue') in COLS else ('NetAfterCb' if 'NetAfterCb' in COLS else None)
                if not rev_col:
                    return {"reply":"No revenue-like column detected in dataset.", "table_html":"", "plot_data_uri": None}
                sql = f"""
                    SELECT STRFTIME('%Y-%m',{date_col}) AS month,
                           SUM(CAST({rev_col} AS DOUBLE)) AS total_revenue
                    FROM {TABLE}
                    WHERE {rel['filter_sql']}
                    GROUP BY month
                    ORDER BY month;
                """
                try:
                    df = run_sql_and_fetch_df(sql)
                except Exception as e:
                    _log_and_mask_error(e, "monthly_revenue_sql_exec")
                    return _safe_user_error("I couldn't fetch revenue data for that period. Please try rephrasing.")
                if df is None or df.empty:
                    return {"reply": "No revenue data found for the requested period.", "table_html":"", "plot_data_uri": None}
                df['month'] = df['month'].astype(str)
                table_html = rows_to_html_table(df.to_dict(orient='records'))
                plot_uri = plot_to_base64(df['month'].tolist(), df['total_revenue'].fillna(0).tolist(), kind='line', title=f"Monthly revenue")
                explanation = f"Month-wise revenue for the selected period (values are total revenue per month)."
                resp = {"reply": explanation, "table_html": table_html, "plot_data_uri": plot_uri, "rows_returned": len(df)}
                if DEBUG_SQL:
                    resp["debug_sql"] = sql
                return resp
            # otherwise ask for year explicitly
            return {"reply": "Please specify the year for which you want month-wise revenue (for example: '2024' or 'July 2024').", "table_html": "", "plot_data_uri": None}

        # explicit year handling
        date_col = DETECTED.get('date') or 'SaleDate'
        rev_col = DETECTED.get('revenue') if DETECTED.get('revenue') in COLS else ('NetAfterCb' if 'NetAfterCb' in COLS else None)
        if not rev_col:
            return {"reply":"No revenue-like column detected in dataset.", "table_html":"", "plot_data_uri": None}

        sql = f"""
            SELECT STRFTIME('%Y-%m',{date_col}) AS month,
                   SUM(CAST({rev_col} AS DOUBLE)) AS total_revenue
            FROM {TABLE}
            WHERE STRFTIME('%Y',{date_col}) = '{year}'
            GROUP BY month
            ORDER BY month;
        """
        try:
            df = run_sql_and_fetch_df(sql)
        except Exception as e:
            _log_and_mask_error(e, "monthly_revenue_sql_exec_year")
            return _safe_user_error("I couldn't fetch revenue data for that year. Please try rephrasing.")
        if df is None or df.empty:
            return {"reply": f"No revenue data found for {year}.", "table_html":"", "plot_data_uri": None}
        df['month'] = df['month'].astype(str)
        table_html = rows_to_html_table(df.to_dict(orient='records'))
        plot_uri = plot_to_base64(df['month'].tolist(), df['total_revenue'].fillna(0).tolist(), kind='line', title=f"Monthly revenue {year}")
        explanation = f"Month-wise revenue for {year} (values are total revenue per month)."
        resp = {"reply": explanation, "table_html": table_html, "plot_data_uri": plot_uri, "rows_returned": len(df)}
        if DEBUG_SQL:
            resp["debug_sql"] = sql
        return resp
    except Exception as e:
        return _safe_user_error(_log_and_mask_error(e, "monthly_revenue_last_year"))


def handle_top_by_entity(question, entity_type='package'):
    """
    Generic handler: entity_type in ('package','campaign','channel','marketing_source','portal_campaign').
    Supports metrics: installations, installation_rate, orders, revenue, profit.
    Accepts optional month/year (strict: month alone asks for year) or relative phrases handled by handle_relative_time.
    """
    try:
        guard = _last_year_guard(question)
        if guard:
            return guard

        # small helper to quote identifiers safely for DuckDB (assumes name from schema)
        def qident(name: str) -> str:
            return f"\"{name.replace('\"', '\"\"')}\""

        ql = (question or "").lower()

        # detect limit
        m = re.search(r'\btop\s+(\d{1,2})\b', ql)
        limit = int(m.group(1)) if m else 10

        # entity -> column mapping via DETECTED or fallbacks
        ent_col = None
        if entity_type == 'package':
            ent_col = DETECTED.get('package') or find_best(COLS, ['Package', 'Product', 'Plan', 'Offering'])
        elif entity_type == 'campaign':
            ent_col = choose_marketing_column(question) or DETECTED.get('campaign') or find_best(COLS, ['Campaign', 'MarketingCampaign', 'CampaignName', 'UtmCampaign'])
        elif entity_type == 'marketing_source':
            ent_col = DETECTED.get('marketing_source')
        elif entity_type == 'portal_campaign':
            ent_col = DETECTED.get('portal_campaign')
        elif entity_type == 'channel':
            ent_col = DETECTED.get('channel') or find_best(COLS, CHANNEL_CANDS)
        if not ent_col:
            return {"reply": f"I couldn't find a column for '{entity_type}' in the dataset. Try using the exact column name (e.g., 'campaign' or 'package').", "table_html": "", "plot_data_uri": None}

        # metric detection
        metric = None
        if re.search(r'\b(install|installation|installs|installed)\b', ql):
            metric = 'installations'
        if re.search(r'\brate\b.*install', ql) or re.search(r'installation\s+rate', ql):
            metric = 'installation_rate'
        if re.search(r'\borders?\b|\bnumber of orders\b', ql):
            metric = 'orders'
        if re.search(r'\brevenue\b|\bsales\b|\bamount\b', ql) and (DETECTED.get('revenue') in COLS):
            metric = 'revenue'
        if re.search(r'\bprofit\b|\bprofits\b|\bnet profit\b', ql):
            metric = 'profit'

        # fallback: if none matched prefer installations if install-related words present
        if not metric:
            if re.search(r'\binstall', ql):
                metric = 'installations'
            else:
                return {"reply": "Please say which metric you want (installations, installation rate, orders, revenue, or profit). Example: 'top campaigns by revenue in July 2025'.", "table_html": "", "plot_data_uri": None}

        # ========== FIXED: INITIALIZE VARIABLES FIRST ==========
        date_col = DETECTED.get('date') or 'SaleDate'
        date_ident = qident(date_col)
        where_clauses = []
        period_text = ""
        
        # exclude null/empty entity values by default (use quoted identifier)
        ent_ident = qident(ent_col)
        where_clauses.append(f"({ent_ident} IS NOT NULL AND TRIM(COALESCE({ent_ident},'')) <> '')")

        # ========== FIXED: SINGLE TIME FILTER LOGIC ==========
        # Check for relative time phrases FIRST (like "last month", "yesterday")
        rel = handle_relative_time(question, sale_date_col=date_col)
        
        if rel and rel.get('action') == 'filter':
            # Use the relative time filter directly
            where_clauses.append(rel['filter_sql'])
            period_text = " for the selected period"
        else:
            # Fall back to explicit month/year parsing
            month, year = parse_month_year_from_text(question)
            
            # Detect explicit year-only (e.g. "2025") if user didn't provide month
            if not year:
                m_year_only = re.search(r'\b(20\d{2})\b', question)
                if m_year_only:
                    year = int(m_year_only.group(1))

            # if month without year -> ask for year (strict policy)
            if month and not year:
                return {"reply": f"Please specify the year for '{calendar.month_name[month]}' (e.g., 'July 2025').", "table_html": "", "plot_data_uri": None}

            if month and year:
                # month + year filter
                last_day = calendar.monthrange(year, month)[1]
                start = date(year, month, 1)
                end = date(year, month, last_day)
                where_clauses.append(f"CAST({date_ident} AS DATE) BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'")
                period_text = f" for {calendar.month_name[month]} {year} (from {start.isoformat()} to {end.isoformat()})"
            elif year and not month:
                # year-only filter
                where_clauses.append(f"STRFTIME('%Y',{date_ident}) = '{year}'")
                period_text = f" for {year}"
            else:
                period_text = ""
        # ========== END FIXED TIME FILTER LOGIC ==========

        # installation detection expression (adjust if your data uses different columns)
        install_condition = "(install_date_parsed IS NOT NULL OR InstallDate IS NOT NULL OR IsInstalled = 1)"

        # metric -> SQL select expression
        if metric == 'installations':
            agg_expr = f"SUM(CASE WHEN {install_condition} THEN 1 ELSE 0 END) AS installations"
            order_by = "installations DESC"
        elif metric == 'installation_rate':
            agg_expr = (
                f"SUM(CASE WHEN {install_condition} THEN 1 ELSE 0 END) AS installations, "
                f"COUNT(*) AS total_orders, "
                f"(CASE WHEN COUNT(*)>0 THEN 100.0 * SUM(CASE WHEN {install_condition} THEN 1 ELSE 0 END) / COUNT(*) ELSE 0 END) AS installation_rate"
            )
            order_by = "installation_rate DESC"
        elif metric == 'orders':
            agg_expr = "COUNT(*) AS orders"
            order_by = "orders DESC"
        elif metric == 'revenue':
            rev_col = DETECTED.get('revenue')
            if not rev_col or rev_col not in COLS:
                return {"reply": "No revenue-like column detected in dataset.", "table_html": "", "plot_data_uri": None}
            rev_ident = qident(rev_col)
            agg_expr = f"SUM(CAST({rev_ident} AS DOUBLE)) AS total_revenue"
            order_by = "total_revenue DESC"
        elif metric == 'profit':
            profit_col = get_profit_col()
            if not profit_col or profit_col not in COLS:
                return {"reply": "No profit-like column detected in dataset.", "table_html": "", "plot_data_uri": None}
            profit_ident = qident(profit_col)
            agg_expr = f"SUM(CAST({profit_ident} AS DOUBLE)) AS total_profit"
            order_by = "total_profit DESC"
        else:
            return {"reply": "Unknown metric requested.", "table_html": "", "plot_data_uri": None}

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        sql = f"""
            SELECT {ent_ident} AS entity,
                   {agg_expr}
            FROM {TABLE}
            WHERE {where_sql}
            GROUP BY {ent_ident}
            ORDER BY {order_by}
            LIMIT {limit};
        """

        try:
            df = run_sql_and_fetch_df(sql)
        except Exception as e:
            _log_and_mask_error(e, "handle_top_by_entity_sql_exec")
            return _safe_user_error("I couldn't compute the requested report right now. Please try rephrasing or ask for a simpler filter.")

        if df is None or df.empty:
            return {"reply": f"No data found{period_text}.", "table_html": "", "plot_data_uri": None}

        # format table & explanation
        # choose best displayed column for metric
        display_metric_col = None
        if metric == 'installations':
            display_metric_col = 'installations'
        elif metric == 'installation_rate':
            display_metric_col = 'installation_rate'
        elif metric == 'orders':
            display_metric_col = 'orders'
        elif metric == 'revenue':
            display_metric_col = 'total_revenue'
        elif metric == 'profit':
            display_metric_col = 'total_profit'

        # ensure column present and fallback safely
        if display_metric_col and display_metric_col not in df.columns:
            nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if nums:
                display_metric_col = nums[0]
            elif df.shape[1] > 1:
                display_metric_col = df.columns[1]
            else:
                display_metric_col = df.columns[0]

        table_html = rows_to_html_table(df.to_dict(orient='records'))
        rows = df.to_dict(orient='records')
        top_list = []
        for r in rows:
            val = r.get(display_metric_col)
            if isinstance(val, float):
                val_str = f"{val:,.2f}"
            elif isinstance(val, (int, np.integer)):
                val_str = f"{int(val):,}"
            else:
                val_str = str(val)

            top_list.append(f"{r.get('entity') or '<NULL>'}: {val_str}")

        explanation = (
            f"Top {len(rows)} {entity_type}s by {metric.replace('_', ' ')}{period_text}:\n"
            + "; ".join(top_list)
        )

        resp = {
            "reply": explanation,
            "table_html": table_html,
            "plot_data_uri": None,
            "rows_returned": len(rows)
        }
        if DEBUG_SQL:
            resp["debug_sql"] = sql
        return resp

    except Exception as e:
        return _safe_user_error(_log_and_mask_error(e, "handle_top_by_entity"))

def handle_compare_orders_between_years(question):
    try:
        years = re.findall(r"\b(20\d{2})\b", question)
        if len(years) >= 2:
            y1, y2 = int(years[0]), int(years[1])
        else:
            # ask user to specify the two years explicitly
            return {"reply": "Please specify the two years to compare (for example: 'Compare orders 2022 and 2023').", "table_html": "", "plot_data_uri": None}

        date_col = DETECTED.get('date') or 'SaleDate'

        sql_counts = f"""
            SELECT STRFTIME('%Y',{date_col}) AS yr,
                   COUNT(*) AS orders,
                   SUM(CAST(COALESCE({DETECTED.get('revenue') or 'NetAfterCb'}, 0) AS DOUBLE)) AS revenue,
                   SUM(CASE WHEN CancelDate IS NOT NULL THEN 1 ELSE 0 END) AS cancels
            FROM {TABLE}
            WHERE STRFTIME('%Y',{date_col}) IN ('{y1}','{y2}')
            GROUP BY yr
            ORDER BY yr;
        """
        try:
            df = run_sql_and_fetch_df(sql_counts)
        except Exception as e:
            _log_and_mask_error(e, "compare_orders_sql_exec")
            return _safe_user_error("I couldn't compute the comparison right now. Please try again later or rephrase the question.")
        if df is None or df.empty:
            return {"reply": f"Sorry — I couldn't find order data for {y1} or {y2}.", "table_html":"", "plot_data_uri": None}

        df['yr'] = df['yr'].astype(str)
        stats = {int(r['yr']): {"orders": int(r['orders']), "revenue": float(r.get('revenue') or 0), "cancels": int(r.get('cancels') or 0)} for r in df.to_dict(orient='records')}

        if y1 not in stats or y2 not in stats:
            return {"reply": f"Data missing for one of the years ({y1}, {y2}).", "table_html":"", "plot_data_uri": None}

        o1 = stats[y1]['orders']
        o2 = stats[y2]['orders']
        r1 = stats[y1]['revenue']
        r2 = stats[y2]['revenue']
        c1 = stats[y1]['cancels']
        c2 = stats[y2]['cancels']

        pct_change = None
        if o1 > 0:
            pct_change = (o2 - o1) / o1 * 100.0
            pct_text = f"{pct_change:.1f}%"
        else:
            pct_text = "N/A"

        reasons = []
        if o2 < o1:
            reasons.append(f"Orders decreased from {o1:,} in {y1} to {o2:,} in {y2} ({pct_text}).")
        else:
            reasons.append(f"Orders changed from {o1:,} in {y1} to {o2:,} in {y2} ({pct_text}).")

        if r2 < r1:
            reasons.append(f"Total revenue also fell from ${r1:,.0f} to ${r2:,.0f}.")
        elif r2 > r1:
            reasons.append(f"Revenue changed from ${r1:,.0f} to ${r2:,.0f} (increase).")

        if c2 > c1:
            reasons.append(f"Cancellations increased ({c1:,} → {c2:,}), which may explain some of the order decline.")

        aov1 = r1 / o1 if o1 else 0
        aov2 = r2 / o2 if o2 else 0
        if aov2 < aov1:
            reasons.append("Average order value decreased, suggesting smaller orders on average.")
        elif aov2 > aov1:
            reasons.append("Average order value increased.")

        explanation = " ".join(reasons[:3])
        summary_df = pd.DataFrame([
            {"year": y1, "orders": o1, "revenue": r1, "cancels": c1},
            {"year": y2, "orders": o2, "revenue": r2, "cancels": c2},
        ])
        table_html = rows_to_html_table(summary_df.to_dict(orient='records'))
        resp = {"reply": explanation, "table_html": table_html, "plot_data_uri": None, "rows_returned": len(summary_df)}
        if DEBUG_SQL:
            resp["debug_sql"] = sql_counts
        return resp
    except Exception as e:
        return _safe_user_error(_log_and_mask_error(e, "compare_orders_between_years"))

# -------------------------
# SQL Safety and LLM Functions
# -------------------------

def is_safe_select(sql: str) -> bool:
    """Check if SQL is a safe SELECT query."""
    sql_clean = re.sub(r"--.*$", "", sql, flags=re.MULTILINE).strip()
    sql_clean = re.sub(r"/\*.*?\*/", "", sql_clean, flags=re.DOTALL).strip()
    first_word = sql_clean.split()[0].upper() if sql_clean else ""
    return first_word == "SELECT"

def normalize_sql_columns(sql: str) -> str:
    """Normalize column names in SQL to match detected columns."""
    # This is a simplified version - you may need to expand this
    # based on your specific column naming needs
    return sql

def ask_model_for_sql(question: str, schema_cols: list, sample_rows: pd.DataFrame) -> str:
    """Ask LLM to generate SQL for the question."""
    try:
        schema_info = f"Columns: {', '.join(schema_cols)}"
        sample_info = sample_rows.head(10).to_csv(index=False) if not sample_rows.empty else "No sample data"
        
        prompt = f"""
        Given the database schema and sample data, generate a SQL query to answer the question.
        Use table name: {TABLE}
        
        {schema_info}
        
        Sample data:
        {sample_info}
        
        Question: {question}
        
        Return only the SQL query without any explanation or markdown formatting.
        If you cannot generate a safe SELECT query, return "NO_SQL".
        """
        
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        sql = response.choices[0].message.content.strip()
        return sql if is_safe_select(sql) else "NO_SQL"
        
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return "NO_SQL"

def ask_model_fix_sql(bad_sql: str, error: str, schema_cols: list, sample_rows: pd.DataFrame) -> str:
    """Ask LLM to fix problematic SQL."""
    try:
        prompt = f"""
        The following SQL query failed with error: {error}
        
        Broken SQL: {bad_sql}
        
        Schema: {', '.join(schema_cols)}
        
        Please fix the SQL query and return only the corrected SQL without any explanation.
        If you cannot fix it safely, return "NO_SQL".
        """
        
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        fixed_sql = response.choices[0].message.content.strip()
        return fixed_sql if is_safe_select(fixed_sql) else "NO_SQL"
        
    except Exception as e:
        print(f"Error fixing SQL: {e}")
        return "NO_SQL"

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, static_folder='static')

SESSION_COOKIE_NAME = "sid"
SESSION_COOKIE_TTL_DAYS = 365 * 2

def _make_sid():
    return uuid.uuid4().hex

@app.before_request
def ensure_sid_cookie():
    sid_from_cookie = request.cookies.get(SESSION_COOKIE_NAME, "").strip()
    sid_from_query = (request.args.get('sid') or "").strip()
    sid_from_payload = ""
    try:
        sid_from_payload = (request.get_json(silent=True) or {}).get('sid') or ""
    except Exception:
        sid_from_payload = ""
    sid = sid_from_payload or sid_from_query or sid_from_cookie or _make_sid()
    g.sid = sid
    g.set_sid_cookie = (sid != sid_from_cookie)

@app.after_request
def set_sid_cookie(response):
    try:
        if getattr(g, "set_sid_cookie", False):
            secure_flag = True
            try:
                if app.debug or request.host.startswith("localhost"):
                    secure_flag = False
            except Exception:
                secure_flag = True
            samesite_val = 'None' if secure_flag else 'Lax'
            response.set_cookie(
                SESSION_COOKIE_NAME,
                g.sid,
                max_age=SESSION_COOKIE_TTL_DAYS * 24 * 3600,
                httponly=True,
                samesite=samesite_val,
                secure=secure_flag,
                path="/"
            )
            try:
                response.headers['X-Debug-SID'] = g.sid
            except Exception:
                pass
    except Exception:
        pass
    return response

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    payload = request.json or {}
    question = (payload.get('message') or "").strip()
    qlow = question.lower()

    if not question:
        return jsonify({"error":"No message provided"}), 400

    # ========== DEBUGGING - ADD THIS ==========
    print(f"DEBUG: Question='{question}'")
    print(f"DEBUG: qlow='{qlow}'")
    # ========== END DEBUGGING ==========
    # DETAILS / PERFORMANCE / SUMMARY QUERIES
    if re.search(r'\b(details?|performance|summary)\b', qlow) and (
        "package" in qlow
        or "marketing source" in qlow
        or "portal campaign" in qlow
        or "main channel" in qlow
        or re.search(r'\bchannel\b', qlow)
    ):
        return jsonify(handle_entity_details(question))

    # 1) Explicit top marketing source / portal campaign
    if re.search(r'\btop\b.*\bmarketing source', qlow):
        try:
            return jsonify(handle_top_by_entity(question, entity_type='marketing_source'))
        except Exception as e:
            return jsonify(_safe_user_error(_log_and_mask_error(e, "top_marketing_source_dispatch"))), 500

    if re.search(r'\btop\b.*\bportal campaign', qlow):
        try:
            return jsonify(handle_top_by_entity(question, entity_type='portal_campaign'))
        except Exception as e:
            return jsonify(_safe_user_error(_log_and_mask_error(e, "top_portal_campaign_dispatch"))), 500

    # 2) FIRST check for channel-specific queries
    if re.search(r'\btop\b.*\b(channel|mainchannel|saleschannel)\b', qlow):
        try:
            return jsonify(handle_top_by_entity(question, entity_type='channel'))
        except Exception as e:
            return jsonify(_safe_user_error(_log_and_mask_error(e, "top_channels_dispatch"))), 500

    # 3) Top packages by installs
    if re.search(r'\btop\b.*\bpackages?\b.*\binstall', qlow) or re.search(r'\btop\b.*\bpackages?\b.*\binstallation', qlow):
        try:
            return jsonify(handle_top_packages_by_installations(question))
        except Exception as e:
            return jsonify(_safe_user_error(_log_and_mask_error(e, "top_packages_dispatch"))), 500

    # 4) SPECIFIC MARKETING PATTERNS - ADDED
    if re.search(r'\btop\b.*\b(marketing|campaign)\b.*\binstall', qlow):
        try:
            return jsonify(handle_top_marketing_campaigns_by_installation(question))
        except Exception as e:
            return jsonify(_safe_user_error(_log_and_mask_error(e, "top_marketing_install_dispatch"))), 500

    if re.search(r'\btop\b.*\b(marketing|campaign)\b.*\bprofit', qlow):
        try:
            if re.search(r'\bmarketing source', qlow):
                return jsonify(handle_top_by_entity(question, entity_type='marketing_source'))
            if re.search(r'\bportal campaign', qlow):
                return jsonify(handle_top_by_entity(question, entity_type='portal_campaign'))
            return jsonify(handle_top_by_entity(question, entity_type='campaign'))
        except Exception as e:
            return jsonify(_safe_user_error(_log_and_mask_error(e, "top_marketing_profit_dispatch"))), 500

    # 5) Top packages by profit
    # match 'package' or 'packages' (case-insensitive already due to qlow)
    if re.search(r'\btop\b.*\bpackages?\b.*\bprofit', qlow):
        try:
            return jsonify(handle_top_packages_by_profit(question))
        except Exception as e:
            return jsonify(_safe_user_error(_log_and_mask_error(e, "top_packages_profit_dispatch"))), 500

    # 6) Generic top by entity handler
    if re.search(r'\btop\b.*\b(campaign|campaigns|marketing|package|packages|channel|channels|mainchannel)\b', qlow):
        try:
            if re.search(r'\bchannel\b|\bmainchannel\b', qlow):
                return jsonify(handle_top_by_entity(question, entity_type='channel'))
            elif re.search(r'\bpackage\b', qlow):
                return jsonify(handle_top_by_entity(question, entity_type='package'))
            elif re.search(r'\bmarketing source', qlow):
                return jsonify(handle_top_by_entity(question, entity_type='marketing_source'))
            elif re.search(r'\bportal campaign', qlow):
                return jsonify(handle_top_by_entity(question, entity_type='portal_campaign'))
            elif re.search(r'\bcampaign\b|\bmarketing\b', qlow):
                return jsonify(handle_top_by_entity(question, entity_type='campaign'))
        except Exception as e:
            return jsonify(_safe_user_error(_log_and_mask_error(e, "top_by_entity_dispatch"))), 500

    # Month-wise revenue
    if re.search(r'\bmonth\b.*\brevenue\b.*(last year|\b202\d\b|\b20\d{2}\b)', qlow) or re.search(r'\bmonthly revenue\b', qlow):
        try:
            return jsonify(handle_monthly_revenue_last_year(question))
        except Exception as e:
            return jsonify(_safe_user_error(_log_and_mask_error(e, "monthly_revenue_dispatch"))), 500

    # Year-to-year orders comparison
    if re.search(r'\bwhy\b.*\b(decrease|decline|drop)\b.*\border', qlow) or re.search(r'\bdecrease of orders\b', qlow):
        try:
            return jsonify(handle_compare_orders_between_years(question))
        except Exception as e:
            return jsonify(_safe_user_error(_log_and_mask_error(e, "compare_orders_dispatch"))), 500

    # show columns
    if re.search(r'\b(show|list|display|give|what are)\b.*\b(columns|fields|attributes|headings)\b', qlow):
        try:
            cols = COLS
            cols_html = "<br>".join(f"{i+1}. <strong>{c}</strong>" for i, c in enumerate(cols))
            return jsonify({
                "reply": f"The dataset contains {len(cols)} columns.",
                "table_html": cols_html,
                "columns": cols
            })
        except Exception as e:
            _log_and_mask_error(e, "fetch_columns")
            return jsonify(_safe_user_error("Could not fetch columns right now.")), 500
    rel = handle_relative_time(question, sale_date_col=(DETECTED.get('date') or 'SaleDate'))
    if rel:
        if rel.get("action") == "now":
            return jsonify({"reply": f"Current date & time (Asia/Kolkata): {rel['text']}", "table_html": "", "plot_data_uri": None})
        if rel.get("action") == "ask_year":
            return jsonify({"reply": rel.get("message"), "table_html": "", "plot_data_uri": None})
        if rel.get("action") == "filter":
            ql = question.lower()
            rev_col = DETECTED.get('revenue') if DETECTED.get('revenue') in COLS else ('NetAfterCb' if 'NetAfterCb' in COLS else None)
            profit_col = get_profit_col()
            date_col = DETECTED.get('date') or 'SaleDate'

            # ---- 1) PROFIT FIRST ----
            if any(tok in ql for tok in ("profit", "profits", "net profit", "total profit")):
                if not profit_col:
                    return jsonify({"reply":"No profit-like column detected in dataset.", "table_html":"", "plot_data_uri": None})
                sql = f"""
                    SELECT STRFTIME('%Y-%m',{date_col}) AS month,
                        SUM(CAST({profit_col} AS DOUBLE)) AS profit,
                        COUNT(*) AS orders
                    FROM {TABLE}
                    WHERE {rel['filter_sql']}
                    GROUP BY month
                    ORDER BY month;
                """
                try:
                    df = run_sql_and_fetch_df(sql)
                    if df is None or df.empty:
                        return jsonify({"reply": "No profit found for the selected period.", "table_html":"", "plot_data_uri": None})
                    table_html = rows_to_html_table(df.to_dict(orient='records')) if not df.empty else "<p><i>No data</i></p>"
                    plot_uri = None
                    if not df.empty:
                        plot_uri = plot_to_base64(df['month'].astype(str).tolist(), df['profit'].fillna(0).tolist(), kind='line', title='Profit over months')
                    resp = {"reply": f"Profit details for the selected period", "table_html": table_html, "plot_data_uri": plot_uri}
                    if DEBUG_SQL:
                        resp["debug_sql"] = sql
                    return jsonify(resp)
                except Exception as e:
                    _log_and_mask_error(e, "filtered_profit")
                    return jsonify(_safe_user_error("I couldn't get profit details for that filter.")), 500

            # ---- 2) REVENUE (NO 'total' TOKEN) ----
            if any(tok in ql for tok in ("revenue", "sales", "amount")):
                if not rev_col:
                    return jsonify({"reply":"No revenue-like column detected in dataset.", "table_html":"", "plot_data_uri": None})
                sql = f"""
                    SELECT STRFTIME('%Y-%m',{date_col}) AS month,
                           SUM(CAST({rev_col} AS DOUBLE)) AS revenue,
                           COUNT(*) AS orders
                    FROM {TABLE}
                    WHERE {rel['filter_sql']}
                    GROUP BY month
                    ORDER BY month;
                """
                try:
                    df = run_sql_and_fetch_df(sql)
                    if df is None or df.empty:
                        return jsonify({"reply": "No revenue found for the selected period.", "table_html":"", "plot_data_uri": None})
                    df_preview = df.head(MAX_ROWS) if MAX_ROWS and len(df) > MAX_ROWS else df
                    table_html = rows_to_html_table(df_preview.to_dict(orient='records')) if not df_preview.empty else "<p><i>No data</i></p>"
                    plot_uri = None
                    if not df.empty:
                        plot_uri = plot_to_base64(df['month'].astype(str).tolist(), df['revenue'].fillna(0).tolist(), kind='line', title=question)
                    explanation = "Here are the revenue results for the selected period."
                    resp = {"reply": explanation, "table_html": table_html, "plot_data_uri": plot_uri, "rows_returned": len(df), "truncated": False}
                    if DEBUG_SQL:
                        resp["debug_sql"] = sql
                    return jsonify(resp)
                except Exception as e:
                    _log_and_mask_error(e, "filtered_revenue")
                    return jsonify(_safe_user_error("I couldn't get the revenue for that filter. Try a simpler question.")), 500

            # ---- 3) INSTALL / DISCONNECT BEFORE ORDERS ----
            install_tokens = ("install", "installation", "disconnection", "disconnect", "installation rate", "disconnection rate")
            if (any(tok in ql for tok in install_tokens) 
                and not re.search(r'\b(marketing|marketting|campaign)\b', ql)):
                print("DEBUG: Running installations relative time query")
                sql = f"""
                    SELECT STRFTIME('%Y-%m',{date_col}) AS month,
                        COUNT(*) AS total_orders,
                        SUM(CASE WHEN (install_date_parsed IS NOT NULL OR InstallDate IS NOT NULL OR IsInstalled = 1) THEN 1 ELSE 0 END) AS installations,
                        SUM(CASE WHEN (DisconnectDate IS NOT NULL OR LOWER(COALESCE(Status,'')) LIKE '%disconnect%') THEN 1 ELSE 0 END) AS disconnections
                    FROM {TABLE}
                    WHERE {rel['filter_sql']}
                    GROUP BY month
                    ORDER BY month;
                """
                try:
                    df = run_sql_and_fetch_df(sql)
                    if df is None or df.empty:
                        return jsonify({"reply": "No install/disconnect info for the selected period.", "table_html":"", "plot_data_uri": None})
                    table_html = rows_to_html_table(df.to_dict(orient='records')) if not df.empty else "<p><i>No data</i></p>"
                    plot_uri = None
                    if not df.empty:
                        plot_uri = plot_to_base64(df['month'].astype(str).tolist(), df['installations'].fillna(0).tolist(), kind='bar', title=question)
                    resp = {"reply": "Installation/disconnection stats for the selected period.", "table_html": table_html, "plot_data_uri": plot_uri, "rows_returned": len(df)}
                    if DEBUG_SQL:
                        resp["debug_sql"] = sql
                    return jsonify(resp)
                except Exception as e:
                    _log_and_mask_error(e, "filtered_install_disconnect")
                    return jsonify(_safe_user_error("I couldn't compute installations/disconnections for that filter.")), 500

            # ---- 4) ORDERS ONLY IF NOT INSTALL QUERY ----
            if any(tok in ql for tok in ("order", "orders", "count")) and not any(tok in ql for tok in install_tokens):
                sql = f"""
                    SELECT STRFTIME('%Y-%m',{date_col}) AS month,
                           COUNT(*) AS orders
                    FROM {TABLE}
                    WHERE {rel['filter_sql']}
                    GROUP BY month
                    ORDER BY month;
                """
                try:
                    df = run_sql_and_fetch_df(sql)
                    if df is None or df.empty:
                        return jsonify({"reply": "No orders for the selected period.", "table_html":"", "plot_data_uri": None})
                    table_html = rows_to_html_table(df.to_dict(orient='records')) if not df.empty else "<p><i>No data</i></p>"
                    resp = {"reply": "Here are the order counts for the selected period.", "table_html": table_html, "plot_data_uri": None, "rows_returned": len(df)}
                    if DEBUG_SQL:
                        resp["debug_sql"] = sql
                    return jsonify(resp)
                except Exception as e:
                    _log_and_mask_error(e, "filtered_orders")
                    return jsonify(_safe_user_error("I couldn't count orders for that filter.")), 500

            # default
            return jsonify({"reply": "Interpreted time filter applied.", "filter_sql": rel['filter_sql']})


    # quick metadata questions
    if re.search(r'\bhow many rows\b|\brow count\b|\bnumber of rows\b|\bhow many columns\b', qlow):
        try:
            rows = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
            cols = len(COLS)
            return jsonify({"reply": f"The dataset contains {rows:,} rows and {cols} columns.", "table_html":"", "plot_data_uri": None})
        except Exception as e:
            _log_and_mask_error(e, "count_rows_cols")
            return jsonify(_safe_user_error("Could not determine rows/cols right now.")), 500

    # Normal handling: ask LLM for SQL
    schema_cols = COLS
    try:
        sample_rows = con.execute(f"SELECT * FROM {TABLE} LIMIT 50").fetchdf()
    except Exception:
        sample_rows = SAMPLE_DF if not SAMPLE_DF.empty else pd.DataFrame(columns=COLS)

    sql_text = ask_model_for_sql(question, schema_cols, sample_rows)

    # Enhanced fallback
    if not sql_text or (isinstance(sql_text, str) and sql_text.strip().upper() == "NO_SQL"):
        ql = qlow

        # 1) campaign / marketing fallback
        if ("campaign" in ql or "marketing" in ql) and (DETECTED.get("campaign") or DETECTED.get("marketing_source") or DETECTED.get("portal_campaign")):
            try:
                return jsonify(handle_top_marketing_campaigns_by_installation(question))
            except Exception as e:
                return jsonify(_safe_user_error(_log_and_mask_error(e, "fallback_top_marketing_campaigns"))), 500

        # 2) package + profit fallback
        if ("package" in ql and "profit" in ql) or ("package" in ql and "by profit" in ql):
            try:
                return jsonify(handle_top_packages_by_profit(question))
            except Exception as e:
                return jsonify(_safe_user_error(_log_and_mask_error(e, "fallback_top_packages_by_profit"))), 500

        # 3) package + install fallback
        if ("package" in ql and ("install" in ql or "installation" in ql)):
            try:
                return jsonify(handle_top_packages_by_installations(question))
            except Exception as e:
                return jsonify(_safe_user_error(_log_and_mask_error(e, "fallback_top_packages_by_installations"))), 500

        # 4) last-resort: try answering from small sample via LLM
        sample_csv = sample_rows.head(20).to_csv(index=False) if not sample_rows.empty else "no sample rows"
        try:
            prompt = f"Columns: {', '.join(COLS)}\nSample:\n{sample_csv}\nQuestion: {question}\nAnswer concisely and factually using only the sample."
            resp = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role":"system","content":"You are a clear data analyst."},{"role":"user","content":prompt}], temperature=0.2, max_tokens=1000)
            ans_text = (resp.choices[0].message.content or "").strip()
            if ans_text:
                return jsonify({"reply": ans_text, "table_html":"", "plot_data_uri": None})
        except Exception:
            pass

        # Final friendly fallback
        return jsonify({"reply": "I couldn't generate a query for that. Please try rephrasing or ask a simpler question.", "table_html": "", "plot_data_uri": None})

    # If we reach here we have some sql_text to sanitize and run
    sql_text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql_text, flags=re.IGNORECASE).strip()
    sql_text = re.sub(r"\bdataset\b", TABLE, sql_text, flags=re.IGNORECASE)

    # normalize columns
    try:
        sql_text = normalize_sql_columns(sql_text)
    except Exception:
        pass

    if not is_safe_select(sql_text):
        return jsonify({"reply":"Generated SQL blocked for safety. Please rephrase.", "table_html":"", "plot_data_uri": None}), 400

    # execute SQL with one auto-fix attempt
    try_count = 0
    last_error = None
    while try_count < 2:
        try:
            df = run_sql_and_fetch_df(sql_text)
            break
        except Exception as e:
            last_error = str(e)
            if try_count == 0:
                try:
                    fixed = ask_model_fix_sql(sql_text, last_error, schema_cols, sample_rows)
                except Exception:
                    fixed = None
                if fixed and fixed.strip().upper() != "NO_SQL":
                    fixed = re.sub(r"^```(?:sql)?\s*|\s*```$", "", fixed, flags=re.IGNORECASE).strip()
                    fixed = re.sub(r"\bdataset\b", TABLE, fixed, flags=re.IGNORECASE)
                    try:
                        fixed = normalize_sql_columns(fixed)
                    except Exception:
                        pass
                    sql_text = fixed
                    try_count += 1
                    continue
            _log_and_mask_error(last_error, "sql_execution")
            return jsonify(_safe_user_error("I couldn't execute that query. Please try a simpler question or be more specific.")), 500

    total_rows = len(df)
    truncated = False
    df_preview = df
    if MAX_ROWS and total_rows > MAX_ROWS:
        truncated = True
        df_preview = df.head(MAX_ROWS)

    table_html = rows_to_html_table(df_preview.to_dict(orient='records')) if not df_preview.empty else "<p><i>No data</i></p>"
    plot_uri = None
    try:
        if not df.empty:
            if 'month' in df.columns:
                nums = [c for c in df.columns if c != 'month' and pd.api.types.is_numeric_dtype(df[c])]
                if nums:
                    ycol = nums[0]
                    plot_uri = plot_to_base64(df['month'].astype(str).tolist(), df[ycol].fillna(0).tolist(), kind='line', title=question)
            else:
                if df.shape[1] >= 2 and pd.api.types.is_numeric_dtype(df.iloc[:,1]):
                    labels = df.iloc[:,0].astype(str).tolist()
                    vals = df.iloc[:,1].astype(float).fillna(0).tolist()
                    plot_uri = plot_to_base64(labels, vals, kind='bar', title=question)
    except Exception as e:
        print("Plot error:", e, traceback.format_exc())
        plot_uri = None

    explanation = ""
    if not df_preview.empty:
        preview_rows = df_preview.head(50).to_dict(orient='records')
        try:
            prompt = f"Question: {question}\nSQL: {sql_text}\nResult preview (first 50 rows JSON):\n{json.dumps(preview_rows, ensure_ascii=False)}\nProvide a 1-line summary and 1-2 sentence factual explanation."
            resp = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role":"system","content":"You are a concise data analyst."},{"role":"user","content":prompt}], temperature=0.0, max_tokens=256)
            explanation = resp.choices[0].message.content.strip()
        except Exception:
            explanation = f"Here are the results for: {question}"
    else:
        explanation = "No data found for that query."

    resp = {
        "reply": explanation,
        "table_html": table_html,
        "plot_data_uri": plot_uri,
        "rows_returned": total_rows,
        "truncated": truncated
    }
    if DEBUG_SQL:
        resp["debug_sql"] = sql_text
    return jsonify(resp)

# -------------------------
# Memory Management
# -------------------------
DB_PATH = Path('./data/chat_memory.db')
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _ensure_chats_table_and_migration(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at INTEGER
            );
        """)
        conn.commit()
        cur.execute("PRAGMA table_info(chats);")
        existing_cols = [r[1] for r in cur.fetchall()]
        if 'session_id' not in existing_cols:
            try:
                cur.execute("ALTER TABLE chats ADD COLUMN session_id TEXT;")
                conn.commit()
            except Exception:
                pass
        cur.execute("PRAGMA table_info(chats);")
        existing_cols = [r[1] for r in cur.fetchall()]
        if 'messages_json' not in existing_cols:
            try:
                cur.execute("ALTER TABLE chats ADD COLUMN messages_json TEXT;")
                conn.commit()
            except Exception:
                pass
        try:
            cur.close()
        except Exception:
            pass
    finally:
        conn.close()

def _row_to_conv(row):
    try:
        if len(row) == 4:
            cid, title, created_at, messages_json = row
            session_id = ""
        elif len(row) == 5:
            cid, title, created_at, session_id, messages_json = row
        else:
            cid = row[0]
            title = row[1] if len(row) > 1 else None
            created_at = row[2] if len(row) > 2 else 0
            session_id = row[3] if len(row) > 3 else ''
            messages_json = row[4] if len(row) > 4 else ''
    except Exception:
        try:
            cid = row[0]
        except Exception:
            cid = "unknown"
        title = ""
        created_at = 0
        session_id = ""
        messages_json = ""
    try:
        messages = json.loads(messages_json) if messages_json else []
    except Exception:
        messages = []
    return {
        "id": cid,
        "title": title,
        "created_at": created_at or 0,
        "session_id": session_id,
        "messages": messages
    }

def _load_conversations():
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            try:
                rows = cur.execute("SELECT id, title, created_at, session_id, messages_json FROM chats ORDER BY created_at DESC").fetchall()
            except Exception:
                rows = cur.execute("SELECT id, title, created_at, messages_json FROM chats ORDER BY created_at DESC").fetchall()
        out = {}
        for r in rows:
            conv = _row_to_conv(r)
            out[conv['id']] = {
                "id": conv['id'],
                "title": conv.get("title") or "Chat",
                "created_at": conv.get("created_at", 0),
                "messages": conv.get("messages", [])
            }
        return out
    except Exception as e:
        try:
            if MEMORY_FILE.exists():
                return json.load(open(MEMORY_FILE, 'r', encoding='utf8'))
        except Exception:
            pass
        return {}

def _save_conversation_single(chat_id, title, created_at, messages, session_id=""):
    _ensure_chats_table_and_migration(DB_PATH)
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(chats);")
            existing_cols = [r[1] for r in cur.fetchall()]
            if 'session_id' in existing_cols and 'messages_json' in existing_cols:
                cur.execute(
                    "INSERT OR REPLACE INTO chats (id, title, created_at, session_id, messages_json) VALUES (?, ?, ?, ?, ?)",
                    (chat_id, title, created_at, session_id, json.dumps(messages))
                )
            elif 'messages_json' in existing_cols:
                cur.execute(
                    "INSERT OR REPLACE INTO chats (id, title, created_at, messages_json) VALUES (?, ?, ?, ?)",
                    (chat_id, title, created_at, json.dumps(messages))
                )
            else:
                raise RuntimeError("SQLite schema missing messages_json column")
            conn.commit()
        return True
    except Exception as e:
        try:
            all_conv = {}
            if MEMORY_FILE.exists():
                try:
                    all_conv = json.load(open(MEMORY_FILE, 'r', encoding='utf8'))
                except Exception:
                    all_conv = {}
            all_conv[chat_id] = {
                "id": chat_id,
                "title": title,
                "created_at": created_at,
                "session_id": session_id,
                "messages": messages
            }
            with open(MEMORY_FILE, 'w', encoding='utf8') as f:
                json.dump(all_conv, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

@app.route('/memory/list', methods=['GET'])
def memory_list():
    sid = (getattr(g, "sid", "") or request.args.get('sid') or "").strip()
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(chats);")
            cols = [r[1] for r in cur.fetchall()]
            if sid and 'session_id' in cols:
                rows = cur.execute("SELECT id, title, created_at FROM chats WHERE session_id = ? ORDER BY created_at DESC", (sid,)).fetchall()
            else:
                rows = cur.execute("SELECT id, title, created_at FROM chats ORDER BY created_at DESC").fetchall()
        chats = [{"id": r[0], "title": r[1] or "Chat", "created_at": r[2]} for r in rows]
        return jsonify({"chats": chats})
    except Exception as e:
        try:
            conv = _load_conversations()
            items = []
            for k, v in conv.items():
                if sid:
                    if isinstance(v, dict) and v.get("session_id") and v.get("session_id") != sid:
                        continue
                items.append({"id": k, "title": v.get("title") or "Chat", "created_at": v.get("created_at", 0)})
            items.sort(key=lambda x: x['created_at'] or 0, reverse=True)
            return jsonify({"chats": items})
        except Exception:
            return jsonify({"error": f"Failed to list chats: {e}"}), 500

@app.route('/memory/load', methods=['GET'])
def memory_load():
    cid = (request.args.get('id') or "").strip()
    sid = (getattr(g, "sid", "") or (request.args.get('sid') or "")).strip()
    if not cid:
        return jsonify({"error": "Missing chat ID"}), 400
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(chats);")
            cols = [r[1] for r in cur.fetchall()]
            if sid and 'session_id' in cols:
                row = cur.execute("SELECT id, title, created_at, session_id, messages_json FROM chats WHERE id = ? AND session_id = ?", (cid, sid)).fetchone()
            else:
                row = cur.execute("SELECT id, title, created_at, messages_json FROM chats WHERE id = ?", (cid,)).fetchone()
        if not row:
            conv = _load_conversations()
            conv_row = conv.get(cid)
            if conv_row:
                if sid and conv_row.get("session_id") and conv_row.get("session_id") != sid:
                    return jsonify({"error": "Chat not found"}), 404
                return jsonify(conv_row)
            return jsonify({"error": "Chat not found"}), 404
        conv = _row_to_conv(row)
        if sid and conv.get("session_id") and conv.get("session_id") != sid:
            return jsonify({"error": "Chat not found"}), 404
        return jsonify({"id": conv["id"], "title": conv.get("title") or "Chat", "created_at": conv.get("created_at", 0), "messages": conv.get("messages", [])})
    except Exception as e:
        return jsonify({"error": f"Failed to load chat: {e}"}), 500

@app.route('/memory/save', methods=['POST'])
def memory_save():
    payload = request.get_json(force=True) or {}
    sid = (getattr(g, "sid") or payload.get('sid') or "").strip()
    cid = payload.get('id') or f"chat-{int(time.time() * 1000)}"
    title = payload.get('title') or 'Chat'
    created_at = payload.get('created_at') or int(time.time() * 1000)
    messages = payload.get('messages') or []
    ok = _save_conversation_single(cid, title, created_at, messages, session_id=sid)
    if not ok:
        return jsonify({"error": "Failed to save chat"}), 500
    return jsonify({"ok": True, "id": cid})

@app.route('/memory/delete', methods=['POST'])
def memory_delete():
    payload = request.get_json(force=True) or {}
    cid = payload.get('id')
    sid = (getattr(g, "sid", "") or payload.get('sid') or "").strip()
    if not cid:
        return jsonify({"error": "Missing chat ID"}), 400
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(chats);")
            cols = [r[1] for r in cur.fetchall()]
            if sid and 'session_id' in cols:
                cur.execute("DELETE FROM chats WHERE id = ? AND session_id = ?", (cid, sid))
            else:
                cur.execute("DELETE FROM chats WHERE id = ?", (cid,))
            conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        try:
            if MEMORY_FILE.exists():
                m = json.load(open(MEMORY_FILE, 'r', encoding='utf8'))
                if cid in m:
                    if sid and m[cid].get("session_id") and m[cid].get("session_id") != sid:
                        return jsonify({"error": "Chat not found"}), 404
                    m.pop(cid, None)
                    with open(MEMORY_FILE, 'w', encoding='utf8') as f:
                        json.dump(m, f, ensure_ascii=False, indent=2)
                    return jsonify({"ok": True})
        except Exception:
            pass
        return jsonify({"error": f"Failed to delete chat: {e}"}), 500

if __name__ == "__main__":
    print(f"Starting server on http://localhost:{PORT} — using table {TABLE}")
    app.run(host="0.0.0.0", port=PORT)
