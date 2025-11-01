import os, json, time
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np

# ---- Env-configurable paths ----
LOG_PATH   = os.environ.get("HIPPO_LOG",  "/tmp/hippo.jsonl")
FRAME_PATH = os.environ.get("HIPPO_FRAME","/tmp/hippo_latest.jpg")
CTRL_PATH  = os.environ.get("HIPPO_CTRL", "/tmp/hippo_cmd.json")
BOOK_DIR   = os.environ.get("HIPPO_BOOK", "/tmp/hippo_bookmarks")
SPEC_PATH  = os.environ.get("HIPPO_SPEC", "/tmp/hippo_spec.png")

Path(BOOK_DIR).mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="TheBrain | Hippocampus", layout="wide")
st.title("🧠 TheBrain — Hippocampus Dashboard")

# ------------------ Controls / sidebar ------------------
col0, col1 = st.columns([3, 1])
with col1:
    log_path   = st.text_input("Log file", LOG_PATH)
    frame_path = st.text_input("Frame path", FRAME_PATH)
    spec_path  = st.text_input("Spectrogram path", SPEC_PATH)
    ctrl_path  = st.text_input("Control file", CTRL_PATH)
    book_dir   = st.text_input("Bookmarks dir", BOOK_DIR)
    refresh    = st.slider("Auto-refresh (sec)", 1, 10, 2)
    max_lines  = st.number_input("Max lines to load", min_value=100, max_value=200000, value=10000, step=500)

    st.divider()

    # Language input → feed
    lang_text = st.text_input("Language input (press Send to encode)", "")
    if st.button("📨 Send text"):
        Path(ctrl_path).write_text(json.dumps({"type":"language","text": lang_text}))
        st.toast("Sent language event", icon="📨")

    st.divider()

    # On-demand actions: write commands to CTRL_PATH
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔎 Retrieve now"):
            Path(ctrl_path).write_text(json.dumps({"type": "retrieve_now"}))
            st.toast("Sent: retrieve_now", icon="🔎")
    with c2:
        label_val = st.text_input("Bookmark label", "")
        if st.button("🔖 Bookmark frame"):
            Path(ctrl_path).write_text(json.dumps({"type": "bookmark", "label": label_val}))
            st.toast("Sent: bookmark", icon="📌")

    cool = st.slider("Encode cooldown (s)", 0.0, 3.0, 1.0, 0.1)
    if st.button("💾 Set cooldown"):
        Path(ctrl_path).write_text(json.dumps({"type": "set_cooldown", "seconds": float(cool)}))
        st.toast(f"Set cooldown → {cool:.1f}s", icon="💾")

# ------------------ Helpers ------------------
@st.cache_data(show_spinner=False)
def read_tail(path: str, n: int) -> List[str]:
    p = Path(path)
    if not p.exists(): return []
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
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df["time"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    elif "t" in df.columns:
        df["time"] = pd.to_datetime(df["t"], unit="s", errors="coerce")
    if "time" in df.columns:
        df = df.sort_values("time")
    return df

# ------------------ Main render loop ------------------
placeholder = st.empty()

while True:
    df = load_df(log_path, int(max_lines))

    with placeholder.container():
        if df.empty:
            st.info("No telemetry yet. Run a demo (e.g., multimodal_feed) that calls telemetry.log_event(...).")
        else:
            # ======= Live camera & audio spectrogram + KPIs =======
            top = st.columns([2, 1])
            with top[0]:
                c_left, c_right = st.columns([1, 1])
                with c_left:
                    st.subheader("Live camera")
                    if Path(frame_path).exists():
                        st.image(frame_path, caption="hippo_latest.jpg", width='stretch')
                    else:
                        st.info("Waiting for live frame…")
                with c_right:
                    st.subheader("Audio: Log-spectrogram")
                    if Path(spec_path).exists():
                        st.image(spec_path, caption="hippo_spec.png", width='stretch')
                    else:
                        st.info("Waiting for spectrogram…")

            # KPIs on the right
            writes     = (df["mode"] == "encode").sum() if "mode" in df else 0
            retrieves  = (df["mode"] == "retrieve").sum() if "mode" in df else 0
            mem_size   = int(df.get("memory_size", pd.Series([np.nan])).dropna().iloc[-1]) if "memory_size" in df else np.nan
            pending    = int(df.get("pending_windows", pd.Series([np.nan])).dropna().iloc[-1]) if "pending_windows" in df else np.nan

            with top[1]:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Encodes", writes)
                k2.metric("Retrieves", retrieves)
                k3.metric("Memory size", mem_size if not np.isnan(mem_size) else "-")
                k4.metric("Pending windows", pending if not np.isnan(pending) else "-")

            # ======= Charts & Tables =======
            cols = st.columns([3, 2])

            with cols[0]:
                if {"novelty","time"}.issubset(df.columns):
                    st.subheader("Novelty over time")
                    st.line_chart(df.set_index("time")["novelty"], width='stretch')

                if {"event_modality","time"}.issubset(df.columns):
                    st.subheader("Events timeline")
                    cols_to_show = [c for c in ["time", "event_modality", "mode", "t"] if c in df.columns]
                    if cols_to_show:
                        st.dataframe(df[cols_to_show].tail(50), width='stretch')

            with cols[1]:
                if {"l2_to_pred","time"}.issubset(df.columns):
                    st.subheader("Vision motion proxy (L2 to EMA)")
                    st.line_chart(df.set_index("time")["l2_to_pred"], width='stretch')

                if {"audio_db","time"}.issubset(df.columns):
                    st.subheader("Audio level (dB)")
                    st.line_chart(df.set_index("time")["audio_db"], width='stretch')

                if {"l2_to_pred_audio","time"}.issubset(df.columns):
                    st.subheader("Audio novelty proxy (L2 to EMA)")
                    st.line_chart(df.set_index("time")["l2_to_pred_audio"], width='stretch')

                # Attention + selected window (from retrieve_now)
                if {"selected_window_id","selected_t_window"}.issubset(df.columns):
                    last_r = df[df.get("action","") == "retrieve_now"].tail(1)
                    if not last_r.empty:
                        st.subheader("Last retrieve — selected window")
                        wid = last_r["selected_window_id"].iloc[0]
                        twin = last_r["selected_t_window"].iloc[0]
                        st.write(f"**window_id:** {wid}   |   **t_window:** {twin}")

                if {"attn_indices","attn_scores"}.issubset(df.columns):
                    st.subheader("Last retrieve — attention")
                    last_ret = df[df.get("action","") == "retrieve_now"].tail(1)
                    if not last_ret.empty:
                        idxs = last_ret["attn_indices"].iloc[0]
                        scs  = last_ret["attn_scores"].iloc[0]
                        if isinstance(idxs, list) and isinstance(scs, list) and len(idxs) == len(scs) and len(idxs) > 0:
                            attn_df = pd.DataFrame({"memory_id": idxs, "score": scs}).sort_values("score", ascending=False)
                            st.bar_chart(attn_df.set_index("memory_id")["score"], width='stretch')

                # Cross-modal reconstruction cosines (from retrieve_now)
                recon_cols = [c for c in df.columns if c.startswith("recon/")]
                if recon_cols and "time" in df.columns:
                    st.subheader("Cross-modal reconstruction")
                    rec = df[df.get("action","") == "retrieve_now"]
                    rec = rec.dropna(subset=recon_cols).set_index("time")[recon_cols].tail(200)
                    if not rec.empty:
                        st.line_chart(rec, width='stretch')

            # ======= Bookmarks =======
            with st.expander("Bookmarks"):
                items = sorted(Path(book_dir).glob("book_*.json"))
                if not items:
                    st.write("No bookmarks yet.")
                else:
                    for j in items[-10:][::-1]:
                        try:
                            meta = json.loads(Path(j).read_text())
                        except Exception:
                            continue
                        cols_b = st.columns([1, 3])
                        with cols_b[0]:
                            imgp = meta.get("path")
                            if imgp and Path(imgp).exists():
                                cap = meta.get("label") or Path(imgp).name
                                st.image(imgp, caption=cap, width='stretch')
                        with cols_b[1]:
                            t_val = meta.get("t")
                            mode  = meta.get("mode")
                            nov   = meta.get("novelty")
                            ms    = meta.get("mem_size")
                            lbl   = meta.get("label")
                            st.write(f"**t:** {t_val:.3f}  |  **mode:** {mode}  |  **novelty:** {nov:.3f}")
                            st.write(f"**memory size:** {ms}")
                            if lbl: st.write(f"**label:** {lbl}")

            # ======= Raw tail =======
            with st.expander("Raw tail (last 200 rows)"):
                st.dataframe(df.tail(200), width='stretch')

    time.sleep(refresh)
