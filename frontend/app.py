# frontend/app.py
import os, json, time
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np

LOG_PATH = os.environ.get("HIPPO_LOG", "/tmp/hippo.jsonl")

st.set_page_config(page_title="TheBrain | Hippocampus", layout="wide")

st.title("🧠 TheBrain — Hippocampus Dashboard")

col0, col1 = st.columns([3,1])
with col1:
    log_path = st.text_input("Log file", LOG_PATH)
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
        size = f.tell()
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
    if "ts" in df.columns:
        df["time"] = pd.to_datetime(df["ts"], unit="s")
        df = df.sort_values("time")
    return df

placeholder = st.empty()
last_reload = 0

while True:
    df = load_df(log_path, int(max_lines))
    with placeholder.container():
        if df.empty:
            st.info("No telemetry yet. Run a demo that calls telemetry.log_event(...).")
        else:
            # High-level KPIs
            writes = (df["mode"] == "encode").sum() if "mode" in df else 0
            retrieves = (df["mode"] == "retrieve").sum() if "mode" in df else 0
            mem_size = int(df.get("memory_size", pd.Series([np.nan])).dropna().iloc[-1]) if "memory_size" in df else np.nan
            pending = int(df.get("pending_windows", pd.Series([np.nan])).dropna().iloc[-1]) if "pending_windows" in df else np.nan

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Encodes", writes)
            k2.metric("Retrieves", retrieves)
            k3.metric("Memory size", mem_size if not np.isnan(mem_size) else "-")
            k4.metric("Pending windows", pending if not np.isnan(pending) else "-")

            # Novelty & modes over time
            cols = st.columns([3,2])
            with cols[0]:
                if "novelty" in df and "time" in df:
                    st.subheader("Novelty over time")
                    st.line_chart(df.set_index("time")["novelty"])
                if "event_modality" in df and "time" in df:
                    st.subheader("Events timeline")
                    st.dataframe(df[["time","event_modality","mode","t"]].tail(50), use_container_width=True)
            with cols[1]:
                if set(["attn_indices","attn_scores"]).issubset(df.columns):
                    st.subheader("Last retrieval — attention")
                    last = df[df["mode"] == "retrieve"].tail(1)
                    if not last.empty:
                        idxs = (last["attn_indices"].iloc[0] or [])
                        scs = (last["attn_scores"].iloc[0] or [])
                        attn_df = pd.DataFrame({"memory_id": idxs, "score": scs})
                        attn_df = attn_df.sort_values("score", ascending=False)
                        st.bar_chart(attn_df.set_index("memory_id")["score"])
                # Reconstruction metrics if you log them
                recon_cols = [c for c in df.columns if c.startswith("recon/")]
                if recon_cols:
                    st.subheader("Reconstruction cosines")
                    rec = df.dropna(subset=recon_cols).set_index("time")[recon_cols].tail(200)
                    st.line_chart(rec)

            with st.expander("Raw tail (last 200 rows)"):
                st.dataframe(df.tail(200), use_container_width=True)

    last_reload = time.time()
    time.sleep(refresh)
