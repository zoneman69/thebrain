import os, json, time
from pathlib import Path
from typing import List, Dict, Any
from PIL import UnidentifiedImageError

import streamlit as st
import pandas as pd
import numpy as np

# ---- Env-configurable paths ----
LOG_PATH   = os.environ.get("HIPPO_LOG",  "/tmp/hippo.jsonl")
FRAME_PATH = os.environ.get("HIPPO_FRAME","/tmp/hippo_latest.jpg")
CTRL_PATH  = os.environ.get("HIPPO_CTRL", "/tmp/hippo_cmd.json")
BOOK_DIR   = os.environ.get("HIPPO_BOOK", "/tmp/hippo_bookmarks")
SPEC_PATH  = os.environ.get("HIPPO_SPEC", "/tmp/hippo_spec.png")
REPLAY_DIR_DEFAULT = os.environ.get("PI_REPLAY_DIR", "/home/image/thebrain/replay")

Path(BOOK_DIR).mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="TheBrain | Hippocampus", layout="wide")
st.title("🧠 TheBrain — Hippocampus Dashboard")

# ------------------ Sidebar: Controls / config ------------------
with st.sidebar:
    st.header("Config & controls")

    log_path   = st.text_input("Log file", LOG_PATH)
    frame_path = st.text_input("Frame path", FRAME_PATH)
    spec_path  = st.text_input("Spectrogram path", SPEC_PATH)
    ctrl_path  = st.text_input("Control file", CTRL_PATH)
    book_dir   = st.text_input("Bookmarks dir", BOOK_DIR)
    replay_dir = st.text_input("Replay dir", REPLAY_DIR_DEFAULT)

    auto_refresh = st.toggle(
        "Auto-refresh",
        value=True,
        help="Pause to inspect a single frame/telemetry snapshot.",
    )
    manual_refresh = st.button(
        "🔄 Refresh now",
        help="Render once when auto-refresh is paused.",
    )
    refresh = st.slider("Auto-refresh (sec)", 2, 10, 3)

    max_lines = st.number_input(
        "Max log lines",
        min_value=500,
        max_value=100000,
        value=5000,
        step=500,
        help="Tail this many log lines from the telemetry file."
    )

    st.divider()

    # ---- Language controls ----
    with st.expander("Language encode / retrieve", expanded=True):
        lang_text = st.text_input(
            "Language input",
            "",
            help="Text payload written to HIPPO_CTRL for language encode/retrieve commands.",
        )
        c_lang1, c_lang2 = st.columns(2)
        with c_lang1:
            if st.button(
                "📨 Send text (encode)",
                help="Writes {'type': 'language', 'text': <Language input>} to HIPPO_CTRL.",
            ):
                Path(ctrl_path).write_text(json.dumps({"type": "language", "text": lang_text}))
                st.toast("Sent language event", icon="📨")
        with c_lang2:
            if st.button(
                "🔎 Retrieve with text cue",
                help="Writes {'type': 'retrieve_with_text', 'text': <Language input>} to HIPPO_CTRL.",
            ):
                Path(ctrl_path).write_text(json.dumps({"type": "retrieve_with_text", "text": lang_text}))
                st.toast("Retrieving with text cue…", icon="🔎")

    # ---- Snapshot management ----
    with st.expander("Snapshots", expanded=False):
        snap_dir = Path("/tmp/hippo_snaps")
        snap_dir.mkdir(exist_ok=True)
        snap_path = snap_dir / f"snap_{int(time.time())}.pt"

        if st.button(
            "💾 Save snapshot",
            help="Writes {'type': 'save_snapshot', 'path': <auto-named .pt file>} to HIPPO_CTRL.",
        ):
            Path(ctrl_path).write_text(json.dumps({"type": "save_snapshot", "path": str(snap_path)}))
            st.toast(f"Saving to {snap_path}", icon="💾")

        load_in = st.text_input(
            "Load snapshot path",
            "",
            help="Path used when writing {'type': 'load_snapshot', 'path': <value>} to HIPPO_CTRL.",
        )
        if st.button(
            "📂 Load snapshot",
            help="Writes {'type': 'load_snapshot', 'path': <Load snapshot path>} to HIPPO_CTRL.",
        ) and load_in:
            Path(ctrl_path).write_text(json.dumps({"type": "load_snapshot", "path": load_in}))
            st.toast(f"Loading {load_in}", icon="📂")

    # ---- Memory actions ----
    with st.expander("Memory actions", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "🔎 Retrieve now",
                help="Writes {'type': 'retrieve_now'} to HIPPO_CTRL.",
            ):
                Path(ctrl_path).write_text(json.dumps({"type": "retrieve_now"}))
                st.toast("Sent: retrieve_now", icon="🔎")
        with c2:
            label_val = st.text_input(
                "Bookmark label",
                "",
                help="Label stored when writing {'type': 'bookmark', 'label': <label>} to HIPPO_CTRL.",
            )
            if st.button(
                "🔖 Bookmark frame",
                help="Writes {'type': 'bookmark', 'label': <Bookmark label>} to HIPPO_CTRL.",
            ):
                Path(ctrl_path).write_text(json.dumps({"type": "bookmark", "label": label_val}))
                st.toast("Sent: bookmark", icon="📌")

        cool = st.slider("Encode cooldown (s)", 0.0, 3.0, 1.0, 0.1)
        if st.button(
            "💾 Set cooldown",
            help="Writes {'type': 'set_cooldown', 'seconds': <Encode cooldown>} to HIPPO_CTRL.",
        ):
            Path(ctrl_path).write_text(json.dumps({"type": "set_cooldown", "seconds": float(cool)}))
            st.toast(f"Set cooldown → {cool:.1f}s", icon="💾")

# ------------------ Helpers ------------------
def read_tail(path: str, n: int) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
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

def show_image_safe(path: str, caption: str):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        st.info(f"Waiting for {caption}…")
        return
    # small retry loop to handle very recent writes
    for _ in range(3):
        try:
            with open(p, "rb") as f:
                data = f.read()
            st.image(data, caption=caption, use_column_width=True)
            return
        except UnidentifiedImageError:
            time.sleep(0.1)
        except Exception:
            time.sleep(0.1)
    st.warning(f"Could not render {caption} yet.")

@st.cache_data(ttl=3.0)
def load_df_cached(path: str, n: int) -> pd.DataFrame:
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
        df["time"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    elif "t" in df.columns:
        df["time"] = pd.to_datetime(df["t"], unit="s", errors="coerce")
    if "time" in df.columns:
        df = df.sort_values("time")
    return df

def latest(df: pd.DataFrame, col: str):
    if col not in df:
        return np.nan
    series = pd.Series(df[col]).dropna()
    return series.iloc[-1] if not series.empty else np.nan

def downsample(df: pd.DataFrame, max_points: int = 1000) -> pd.DataFrame:
    if df.empty or len(df) <= max_points:
        return df
    step = max(len(df) // max_points, 1)
    return df.iloc[::step]

@st.cache_data(ttl=5.0)
def summarize_replay_dir(path: str) -> Dict[str, Any]:
    replay_path = Path(path)
    if not replay_path.exists():
        return {"exists": False, "files": 0, "episodes": 0, "last_ts": None}

    files = sorted(replay_path.glob("*.npz"))
    total = 0
    last_ts: float | None = None
    for f in files:
        try:
            payload = np.load(f, allow_pickle=True)
            fused = payload.get("fused")
            if fused is not None:
                total += int(fused.shape[0])
            metadata = payload.get("metadata")
            ts_candidate = None
            if metadata is not None and metadata.size > 0:
                meta = metadata[-1]
                if isinstance(meta, dict):
                    for key in ("t", "ts", "timestamp"):
                        if key in meta:
                            try:
                                ts_candidate = float(meta[key])
                                break
                            except Exception:
                                pass
            ts_candidate = ts_candidate or f.stat().st_mtime
            last_ts = ts_candidate if last_ts is None else max(last_ts, ts_candidate)
        except Exception:
            continue

    return {"exists": True, "files": len(files), "episodes": total, "last_ts": last_ts}

def fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return str(ts)

def file_health(path: str, freshness_sec: float = 5.0) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {"status": "missing", "age": None}
    now = time.time()
    mtime = p.stat().st_mtime
    age = now - mtime
    status = "fresh" if age <= freshness_sec else "stale"
    return {"status": status, "age": age}

# ------------------ Main render (single pass) ------------------
df = load_df_cached(log_path, int(max_lines))
replay_summary = summarize_replay_dir(replay_dir)

# ------------------ Brain vitals ------------------
frame_health = file_health(frame_path, freshness_sec=5.0)
spec_health  = file_health(spec_path,  freshness_sec=5.0)
log_health   = file_health(log_path,   freshness_sec=5.0)

v1, v2, v3, v4 = st.columns(4)

# Camera
frame_label = "OK" if frame_health["status"] == "fresh" else frame_health["status"]
frame_age = (
    f"{frame_health['age']:.1f}s ago"
    if frame_health["age"] is not None
    else "—"
)
v1.metric("Camera frame", frame_label, frame_age)

# Audio
spec_label = "OK" if spec_health["status"] == "fresh" else spec_health["status"]
spec_age = (
    f"{spec_health['age']:.1f}s ago"
    if spec_health["age"] is not None
    else "—"
)
v2.metric("Audio spectrogram", spec_label, spec_age)

# Log
log_label = "OK" if log_health["status"] == "fresh" else log_health["status"]
log_age = (
    f"{log_health['age']:.1f}s ago"
    if log_health["age"] is not None
    else "—"
)
v3.metric("Telemetry log", log_label, log_age)

# Replay
if replay_summary["exists"]:
    last_ts = replay_summary["last_ts"]
    age_str = (
        f"{time.time() - last_ts:.1f}s ago"
        if last_ts is not None
        else "—"
    )
    v4.metric("Episodes recorded", replay_summary["episodes"], age_str)
else:
    v4.metric("Episodes recorded", 0, "no replay dir")

# ------------------ Tabs ------------------
tab_overview, tab_metrics, tab_bookmarks, tab_raw = st.tabs(
    ["Overview", "Metrics", "Bookmarks", "Raw log"]
)

# ======= Overview tab =======
with tab_overview:
    top = st.columns([2, 1])

    # Live camera & audio spectrogram
    with top[0]:
        c_left, c_right = st.columns(2)
        with c_left:
            st.subheader("Live camera")
            show_image_safe(frame_path, "Live frame")
        with c_right:
            st.subheader("Audio: log-spectrogram")
            show_image_safe(spec_path, "Spectrogram")

    with top[1]:
        st.subheader("Replay ingestion")
        if not replay_summary["exists"]:
            st.info(f"Replay dir not found: `{replay_dir}`")
        else:
            rcols = st.columns([2, 1])
            rcols[0].markdown(f"`{replay_dir}`")
            rcols[1].metric("Replay files", replay_summary["files"])
            st.caption(f"Last episode at: {fmt_ts(replay_summary['last_ts'])}")

    if df.empty:
        st.info(
            f"""
No telemetry yet. To get started:
• Run a producer (e.g., `multimodal_feed`) that calls `telemetry.log_event(...)`.
• Verify the expected files are being written:
  - Log: `{log_path}`
  - Frame: `{frame_path}`
  - Spectrogram: `{spec_path}`
  - Control out: `{ctrl_path}`
  - Bookmarks dir: `{book_dir}`
• Checklist for producers:
  - emit `mode` + `event_modality`
  - include `ts`/`t` timestamps
  - optional KPIs: `memory_size`, `pending_windows`, novelty/EMA signals
"""
        )
    else:
        st.subheader("Recent events")
        cols_to_show = [c for c in ["time", "event_modality", "mode", "t"] if c in df.columns]
        if cols_to_show:
            st.dataframe(df[cols_to_show].tail(50), use_container_width=True)

# ======= Metrics tab =======
with tab_metrics:
    if df.empty:
        st.info("No telemetry loaded; metrics unavailable.")
    else:
        # KPIs
        writes     = (df["mode"] == "encode").sum() if "mode" in df else 0
        retrieves  = (df["mode"] == "retrieve").sum() if "mode" in df else 0
        mem_val    = latest(df, "memory_size")
        mem_size   = int(mem_val) if not np.isnan(mem_val) else np.nan
        pending_val = latest(df, "pending_windows")
        pending    = int(pending_val) if not np.isnan(pending_val) else np.nan

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Encodes", writes)
        k2.metric("Retrieves", retrieves)
        k3.metric("Memory size", mem_size if not np.isnan(mem_size) else "-")
        k4.metric("Pending windows", pending if not np.isnan(pending) else "-")

        st.divider()
        cols = st.columns([3, 2])

        with cols[0]:
            # Novelty
            if {"novelty", "time"}.issubset(df.columns):
                st.subheader("Novelty over time")
                nov = downsample(df[["time", "novelty"]].dropna())
                if not nov.empty:
                    st.line_chart(nov.set_index("time")["novelty"])

            # Event timeline (table is lighter than a chart)
            if {"event_modality", "time"}.issubset(df.columns):
                st.subheader("Events timeline (tail)")
                cols_to_show = [c for c in ["time", "event_modality", "mode", "t"] if c in df.columns]
                if cols_to_show:
                    st.dataframe(df[cols_to_show].tail(100), use_container_width=True)

        with cols[1]:
            # Vision motion proxy
            if {"l2_to_pred", "time"}.issubset(df.columns):
                st.subheader("Vision motion proxy (L2 to EMA)")
                tmp = downsample(df[["time", "l2_to_pred"]].dropna())
                if not tmp.empty:
                    st.line_chart(tmp.set_index("time")["l2_to_pred"])

            # Audio level
            if {"audio_db", "time"}.issubset(df.columns):
                st.subheader("Audio level (dB)")
                tmp = downsample(df[["time", "audio_db"]].dropna())
                if not tmp.empty:
                    st.line_chart(tmp.set_index("time")["audio_db"])

            # Audio novelty
            if {"l2_to_pred_audio", "time"}.issubset(df.columns):
                st.subheader("Audio novelty proxy (L2 to EMA)")
                tmp = downsample(df[["time", "l2_to_pred_audio"]].dropna())
                if not tmp.empty:
                    st.line_chart(tmp.set_index("time")["l2_to_pred_audio"])

        st.divider()

        # Attention + selected window (from retrieve actions)
        right_cols = st.columns(2)

        with right_cols[0]:
            if {"selected_window_id", "selected_t_window"}.issubset(df.columns):
                if "action" in df.columns:
                    last_r = df[df["action"].isin(["retrieve_now", "retrieve_with_text"])].tail(1)
                else:
                    last_r = df.tail(1)
                if not last_r.empty:
                    st.subheader("Last retrieve — selected window")
                    wid = last_r["selected_window_id"].iloc[0]
                    twin = last_r["selected_t_window"].iloc[0]
                    st.write(f"**window_id:** {wid}   |   **t_window:** {twin}")

        with right_cols[1]:
            if {"attn_indices", "attn_scores"}.issubset(df.columns):
                if "action" in df.columns:
                    last_ret = df[df["action"].isin(["retrieve_now", "retrieve_with_text"])].tail(1)
                else:
                    last_ret = df.tail(1)
                if not last_ret.empty:
                    st.subheader("Last retrieve — attention")
                    idxs = last_ret["attn_indices"].iloc[0]
                    scs  = last_ret["attn_scores"].iloc[0]
                    if isinstance(idxs, list) and isinstance(scs, list) and len(idxs) == len(scs) and len(idxs) > 0:
                        attn_df = pd.DataFrame({"memory_id": idxs, "score": scs}).sort_values("score", ascending=False)
                        st.bar_chart(attn_df.set_index("memory_id")["score"])

        st.divider()

        # Cross-modal reconstruction cosines (from retrieve actions)
        recon_cols = [c for c in df.columns if c.startswith("recon/")]
        if recon_cols and "time" in df.columns:
            st.subheader("Cross-modal reconstruction")
            if "action" in df.columns:
                rec = df[df["action"].isin(["retrieve_now", "retrieve_with_text"])]
            else:
                rec = df.copy()
            rec = rec.dropna(subset=recon_cols)
            rec = rec.set_index("time")[recon_cols].tail(200)
            if not rec.empty:
                rec = downsample(rec)
                st.line_chart(rec)

# ======= Bookmarks tab =======
with tab_bookmarks:
    st.subheader("Bookmarks")
    items = sorted(Path(book_dir).glob("book_*.json"))
    if not items:
        st.write("No bookmarks yet.")
    else:
        q = st.text_input("Filter by label contains", key="bookmark_filter_query")
        shown = 0
        for j in items[::-1]:
            try:
                meta = json.loads(Path(j).read_text())
            except Exception:
                continue
            lbl = (meta.get("label") or "")
            if q and q.lower() not in lbl.lower():
                continue

            cols_b = st.columns([1, 3])
            with cols_b[0]:
                imgp = meta.get("path")
                if imgp and Path(imgp).exists():
                    cap = lbl or Path(imgp).name
                    st.image(imgp, caption=cap, use_container_width=True)
            with cols_b[1]:
                t_val = meta.get("t")
                mode  = meta.get("mode")
                nov   = meta.get("novelty")
                ms    = meta.get("mem_size")
                t_str = f"{t_val:.3f}" if isinstance(t_val, (int, float)) else t_val
                nov_str = f"{nov:.3f}" if isinstance(nov, (int, float)) else nov
                st.write(f"**t:** {t_str}  |  **mode:** {mode}  |  **novelty:** {nov_str}")
                st.write(f"**memory size:** {ms}")
                if lbl:
                    st.write(f"**label:** {lbl}")

            shown += 1
            if shown >= 10:
                break

        if shown == 0:
            st.write("No bookmarks match filter." if q else "No bookmarks yet.")

# ======= Raw log tab =======
with tab_raw:
    st.subheader("Raw tail (last 200 rows)")
    if df.empty:
        st.write("No telemetry rows yet.")
    else:
        st.dataframe(df.tail(200), use_container_width=True)

# -------- Auto / manual refresh logic --------
if auto_refresh:
    time.sleep(refresh)
    st.rerun()
else:
    if manual_refresh:
        st.rerun()
    # else: page stays static
