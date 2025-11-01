import os, json, time
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np

LOG_PATH = os.environ.get("HIPPO_LOG", "/tmp/hippo.jsonl")
FRAME_PATH = os.environ.get("HIPPO_FRAME", "/tmp/hippo_latest.jpg")

st.set_page_config(page_title="TheBrain | Hippocampus", layout="wide")
st.title("🧠 TheBrain — Hippocampus Dashboard")

col0, col1 = st.columns([3, 1])
with col1:
    log_path = st.text_input("Log file", LOG_PATH)
    frame_path = st.text_input("Frame path", FRAME_PATH)
    refresh = st.slider("Auto-refresh (sec)", 1, 10, 2)
    max_lines = st.number_input("Max lines to load", min_value=100, max_value=200000, value=10000, step=500)

@st.cache_data(show_spinner=False)
def read_tail(path: str, n: int) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    # Simple fast-ish tail
    with open(p, "rb") as f:
        f.seek(0, 2)
        block = 4096
        data = b""
        while len(data.splitlines()) <= n and f.tell() > 0:
            step = min(block, f.tell())
            f.seek(-step, 1)
            data = f.read(step) + data
            f.seek(-step, 1)
        lines = data.splitlines()[-n:]
    return [l.decode("utf-8", errors="ignore") for l in lines]

def load_df(path: str, n: int) -> pd.DataFrame:
    lines = read_tail(path, n)
    rows: List[Dict[str, Any]] = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # add time column if telemetry didn't add ts
    if "ts" in df.columns:
        df["time"] = pd.to_datetime(df["ts"], unit="s")
    else:
        if "t" in df.columns:
            df["time"] = pd.to_datetime(df["t"], unit="s", errors="coerce")
    if "time" in df.columns:
        df = df.sort_values("time")
    return df

placeholder = st.empty()

while True:
    df = load_df(log_path, int(max_lines))
    with placeholder.container():
        if df.empty:
            st.info("No telemetry yet. Run a demo that calls telemetry.log_event(...).")
        else:
            # === Live camera ===
            img_cols = st.columns([2, 1])
            with img_cols[0]:
                st.subheader("Live camera")
                if Path(frame_path).exists():
                    st.image(frame_path, caption="hippo_latest.jpg", width='stretch')
                else:
                    st.info("Waiting for live frame…")

            # === KPIs ===
            writes = (df["mode"] == "encode").sum() if "mode" in df else 0
            retrieves = (df["mode"] == "retrieve").sum() if "mode" in df else 0
            mem_size = int(df.get("memory_size", pd.Series([np.nan])).dropna().iloc[-1]) if "memory_size" in df else np.nan
            pending = int(df.get("pending_windows", pd.Series([np.nan])).dropna().iloc[-1]) if "pending_windows" in df else np.nan

            k1, k2, k3, k4 = img_cols[1].columns(1)
            k1.metric("Encodes", writes)
            k2.metric("Retrieves", retrieves)
            k3.metric("Memory size", mem_size if not np.isnan(mem_size) else "-")
            k4.metric("Pending windows", pending if not np.isnan(pending) else "-")

            # === Charts ===
            cols = st.columns([3, 2])
            with cols[0]:
                if "novelty" in df and "time" in df:
                    st.subheader("Novelty over time")
                    st.line_chart(df.set_index("time")["novelty"], width='stretch')
                if "event_modality" in df and "time" in df:
                    st.subheader("Events timeline")
                    st.dataframe(df[["time", "event_modality", "mode", "t"]].tail(50), width='stretch')

            with cols[1]:
                if "l2_to_pred" in df.columns and "time" in df.columns:
                    st.subheader("L2 to EMA-pred (motion proxy)")
                    st.line_chart(df.set_index("time")["l2_to_pred"], width='stretch')

                # Reconstruction metrics if logged (e.g., demo_decode)
                recon_cols = [c for c in df.columns if c.startswith("recon/")]
                if recon_cols:
                    st.subheader("Reconstruction cosines")
                    rec = df.dropna(subset=recon_cols).set_index("time")[recon_cols].tail(200)
                    st.line_chart(rec, width='stretch')

            with st.expander("Raw tail (last 200 rows)"):
                st.dataframe(df.tail(200), width='stretch')

    time.sleep(refresh)
