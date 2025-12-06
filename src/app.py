#!/usr/bin/env python3
"""Clean Streamlit app for Elite-K phone review analyzer.

This file contains a single consistent implementation and no duplicated
blocks. It imports helpers from src.app_helpers for CSV normalization.
"""

# IMPORTANT: Set this BEFORE importing any HuggingFace libraries
# to suppress subprocess fork warnings when using Ollama CLI
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import re
from typing import Tuple, List, Dict
import difflib

from src.config import DEFAULT_CONFIG
from src import sentiment_engine
from src.api_clients import HFClient, MockHFClient
from src.fake_detector import detect_fake_reviews
from src.sentiment_engine import analyze_sentiments
from src.summarizer import summarize_reviews
from src.video_pipeline import analyze_video, fuse_scores
from src.adapters import get_adapter
from src.score_phone import compute_trust_score
from src.app_helpers import ensure_review_columns, ensure_product_column
from src.aspect_extractor import map_sentences_to_reviews

st.set_page_config(
    page_title="Elite-K Phone Review Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded"  # Open sidebar by default
)
st.title("Elite-K — Smartphone Review Analyzer (MVP)")

# initialize session state keys used for UI persistence
for k, v in {
    "last_results": None,
    "last_df_rows": None,
    "pinned_product": None,
    "selected_product": None,
    "sort_field": "final_score",
    "sort_dir": "Descending",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

    # ensure HF client selection is preserved in session state for reproducible runs
    if 'use_hf' not in st.session_state:
        st.session_state['use_hf'] = False

def detect_price_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Detect a price column name heuristically. Returns (df, price_col_or_None)."""
    lower = {c.lower(): c for c in df.columns}
    candidates = ["price", "cost", "amount", "price_usd", "retail_price", "msrp"]
    for cand in candidates:
        if cand in lower:
            df = df.rename(columns={lower[cand]: "price"})
            # try to coerce the detected column to float
            df["price"] = df["price"].apply(_parse_price)
            return df, "price"
    # fallback: numeric column with plausible price range
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in num_cols:
        vals = df[c].dropna()
        if len(vals) == 0:
            continue
        # check percent of values within 0..1500 USD
        within = ((vals >= 0) & (vals <= 1500)).sum() / len(vals)
        if within > 0.6:
            df = df.rename(columns={c: "price"})
            df["price"] = df["price"].astype(float)
            return df, "price"
    # try to find string columns that look like currency (e.g., "$199", "199.99", "1,299", "199 USD")
    str_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for c in str_cols:
        sample = df[c].dropna().astype(str).head(200)
        if len(sample) == 0:
            continue
        parsed = sample.map(_parse_price)
        success = parsed.notna().sum() / len(parsed)
        if success > 0.6:
            # coerce entire column
            df = df.rename(columns={c: "price"})
            df["price"] = df["price"].map(_parse_price)
            # check plausible range
            vals = df["price"].dropna()
            if len(vals) > 0 and ((vals >= 0) & (vals <= 1500)).sum() / len(vals) > 0.5:
                return df, "price"
            # else undo and continue
            df = df.rename(columns={"price": c})
    return df, None


def _parse_price(x):
    """Parse a price-like value into float, stripping currency symbols and commas. Returns NaN on failure."""
    try:
        if pd.isna(x):
            return float("nan")
    except Exception:
        pass
    s = str(x).strip()
    # remove common currency symbols and letters
    s = re.sub(r"[\$,£€]", "", s)
    s = re.sub(r"(?i)\s*usd\b", "", s)
    s = s.replace(',', '')
    s = s.replace('\u00a0', '')
    s = s.strip()
    # allow values like '199', '199.99', '-'
    if s in ['', '-', '–', '—']:
        return float("nan")
    # extract first numeric occurrence
    m = re.search(r"-?\d{1,3}(?:\d{3})*(?:[\.,]\d+)?|-?\d+(?:[\.,]\d+)?", s)
    if not m:
        return float("nan")
    num = m.group(0)
    num = num.replace(',', '')
    num = num.replace(' ', '')
    try:
        return float(num)
    except Exception:
        try:
            return float(num.replace(',', ''))
        except Exception:
            return float("nan")


def normalize_model_name(name: str) -> str:
    """Normalize model string to a base model name used for grouping.

    Removes memory/storage tokens and parentheses, trims whitespace.
    """
    if name is None:
        return ""
    s = str(name)
    # remove parentheses
    s = re.sub(r"\(.*?\)", "", s)
    # remove RAM/storage tokens like '6GB RAM', '8 GB', '128GB'
    s = re.sub(r"\b\d+\s*(gb|mb|tb)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bram\b", "", s, flags=re.IGNORECASE)
    # remove 'with' clauses and hyphenated trailing specs
    s = re.sub(r"with\b.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b1280p|1080p|4k|5g\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[-–—].*$", "", s)
    # collapse multiple spaces and strip
    s = re.sub(r"\s+", " ", s).strip()
    return s


def group_models(names: List[str], cutoff: float = 0.85) -> Dict[str, List[str]]:
    """Group variant model names under a base normalized model.

    Returns mapping: base_name -> list of original variants
    """
    norm_map = {n: normalize_model_name(n) or n for n in names}
    bases = {}
    used = set()
    unique_norms = list({v for v in norm_map.values()})
    for norm in unique_norms:
        bases[norm] = []
    # fuzzy match originals to norms
    for orig, norm in norm_map.items():
        # use difflib to find closest norm (including itself)
        choices = difflib.get_close_matches(norm, unique_norms, n=1, cutoff=cutoff)
        key = choices[0] if choices else norm
        bases.setdefault(key, []).append(orig)
    # remove empty keys
    return {k: sorted(set(v)) for k, v in bases.items() if v}


def _cols_up_to_price(df: pd.DataFrame) -> list:
    """Return a list of columns up to and including the first 'price' column (case-insensitive).

    If no price column is present, return the original column order.
    """
    if df is None or df.columns is None:
        return []
    cols = list(df.columns)
    lower = [c.lower() for c in cols]
    if 'price' in lower:
        idx = lower.index('price')
        return cols[: idx + 1]
    return cols


def detect_brands(df: pd.DataFrame, product_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """Attempt to infer a `brand` column from the `product` values.

    Returns (df_with_brand_col, sorted_brand_list).
    """
    known = ["Samsung", "Apple", "Xiaomi", "OnePlus", "Oppo", "Vivo", "Google", "Sony", "Motorola", "Nokia", "Realme", "Huawei", "LG", "HTC"]
    brands = set()

    def infer_brand(prod):
        s = str(prod or "").strip()
        if not s:
            return "(unknown)"
        for b in known:
            if re.search(rf"\b{re.escape(b)}\b", s, flags=re.I):
                brands.add(b)
                return b
        # fallback: first token (capitalized) or whole first word
        first = s.split()[0]
        brands.add(first)
        return first

    df = df.copy()
    if product_col in df.columns:
        # store brands normalized to lowercase to avoid case-sensitivity issues
        df["brand"] = df[product_col].fillna("").astype(str).map(lambda p: infer_brand(p).lower())
    else:
        df["brand"] = "(unknown)"
        brands.add("(unknown)")
    return df, sorted({b.title() for b in brands})


def render_phone_table(df: pd.DataFrame, results: Dict[str, Dict]):
    """No-op: phones list section intentionally removed (redundant with the interactive table)."""
    return


def show_product_details(name: str, result: Dict):
    """Show full details for a product result in an expander."""
    with st.expander(f"Details — {name}", expanded=True):
        st.markdown(f"**Final score:** {result.get('trust', {}).get('final_score')}")

        # Friendly per-aspect contribution breakdown (effective weight, aspect score, absolute contribution)
        trust_block = result.get('trust', {}) or {}
        asp_scores = trust_block.get('aspect_scores', {}) or {}
        asp_weights = trust_block.get('aspect_weights_effective', {}) or {}
        asp_contribs = trust_block.get('aspect_contributions', {}) or {}

        # Build a sorted, user-friendly table (sorted by contribution desc)
        contrib_rows = []
        all_asps = sorted(set(list(asp_weights.keys()) + list(asp_scores.keys()) + list(asp_contribs.keys())),
                          key=lambda k: -(asp_contribs.get(k, 0) or 0))
        for asp in all_asps:
            w = asp_weights.get(asp)
            s = asp_scores.get(asp)
            c = asp_contribs.get(asp)
            w_display = f"{w*100:.1f}%" if (w is not None) else "—"
            s_display = f"{s:.2f}" if (s is not None) else "n/a"
            c_display = f"{c:.2f}" if (c is not None) else "n/a"
            contrib_rows.append((asp.replace('_', ' ').title(), w_display, s_display, c_display))

        if contrib_rows:
            try:
                import pandas as _pd
                st.markdown("#### Contributions by aspect")
                st.table(_pd.DataFrame(contrib_rows, columns=["Aspect", "Effective weight", "Aspect score", "Contribution"]))
            except Exception:
                for a, w, s, c in contrib_rows:
                    st.markdown(f"**{a}** — weight: {w}, score: {s}, contribution: {c}")

        st.markdown(f"**Confidence:** {result.get('trust', {}).get('confidence')}")
        st.markdown(f"**Fake reviews:** {result.get('fake_count')} of {result.get('n_total')} ({result.get('fake_pct'):.1f}%)")
        st.subheader("Balanced summary")
        st.write(result.get('summ', {}).get('balanced_summary'))
        # Detailed aspect breakdown as nested expanders (cleaner than raw JSON)
        st.subheader("Aspect breakdown")
        asp_break = result.get('summ', {}).get('aspect_breakdown', {}) or {}
        if asp_break:
            for asp, text in asp_break.items():
                with st.expander(f"{asp.replace('_', ' ').title()}", expanded=False):
                    st.write(text)
        else:
            st.info("No aspect summaries available.")

        # Compact trust details (show key metrics without JSON dump)
        st.subheader("Trust details")
        trust = result.get('trust', {}) or {}
        trust_rows = [
            ("Final score", trust.get('final_score')),
            # Confidence and N reviews intentionally removed for cleaner UI
        ]
        try:
            import pandas as _pd
            st.table(_pd.DataFrame(trust_rows, columns=["metric", "value"]))
        except Exception:
            for k, v in trust_rows:
                st.markdown(f"**{k}:** {v}")



# Sidebar / runtime configuration
st.sidebar.header("Settings")

# Primary action selector moved to the top of the sidebar for clearer UX
tab = st.sidebar.selectbox("Action", ["Demo sample", "Upload CSV", "Analyze YouTube Video"]) 

# Provider selection (moved before mock_mode so it can auto-set the mode)
st.sidebar.subheader("Sentiment Provider")
sentiment_provider = st.sidebar.radio(
    "Choose sentiment analysis provider:",
    ["Mock (fast/deterministic)", "Hugging Face (remote API)", "Ollama (local LLM)"],
    index=0,
    key="sentiment_provider_radio"
)

# Map display name to internal provider key
provider_map = {
    "Mock (fast/deterministic)": "mock",
    "Hugging Face (remote API)": "huggingface",
    "Ollama (local LLM)": "ollama"
}
selected_provider = provider_map[sentiment_provider]

# Auto-set mock_mode based on provider selection
# Mock provider -> mock_mode=True
# HuggingFace/Ollama -> mock_mode=False (use real APIs)
auto_mock_mode = (selected_provider == "mock")

# Fast/local mode removed from sidebar per UX request; default to False.
fast_local_mode = False

# Controls
mock_mode = st.sidebar.checkbox(
    "Mock mode for fake detection & summarization", 
    value=auto_mock_mode,
    help="When enabled, uses fast mock models for fake detection and summarization. Sentiment analysis is controlled by the provider above."
)

# Clear cached results when provider or mock_mode changes
current_settings = f"{selected_provider}_{mock_mode}"
if st.session_state.get('last_settings') != current_settings:
    st.session_state['last_settings'] = current_settings
    if 'last_results' in st.session_state:
        del st.session_state['last_results']
    if 'last_df_rows' in st.session_state:
        del st.session_state['last_df_rows']
    # Show info that settings changed
    if st.session_state.get('settings_changed_once'):
        st.sidebar.info("⚠️ Settings changed! Click 'Analyze selection' to update results.")
    st.session_state['settings_changed_once'] = True

# (Adapter and representative examples hidden for simplified sidebar)
rep_mode = "Review-level (majority)"

# Conditional inputs based on provider
hf_api_key = None
ollama_model = "llama3.2:3b"  # Default fallback

if selected_provider == "huggingface":
    hf_api_key = st.sidebar.text_input("Hugging Face API key (optional)", type="password", key="hf_api_key_input")
    if not hf_api_key and DEFAULT_CONFIG.get("mock_mode", True) is False:
        st.sidebar.warning("Hugging Face API selected but no key provided; will fall back to mock client.")
elif selected_provider == "ollama":
    # === CHANGED: Text input -> Selectbox for easy switching ===
    ollama_model = st.sidebar.selectbox(
        "Select Ollama model:",
        ["llama3.2:3b", "gemma2:2b", "gemma3:1b"],
        index=0,  # Default to llama3.2:3b
        key="ollama_model_select"
    )
    st.sidebar.info(f"Using: **{ollama_model}**")
    # Hint to install if not present
    st.sidebar.caption(f"Command to install: `ollama pull {ollama_model}`")

if getattr(sentiment_engine, 'HF_MODELS_UNAVAILABLE', None):
    st.sidebar.warning("One or more Hugging Face models used by the app are unavailable; the app will use internal mock fallbacks for those models.")

# Debug / advanced (hidden by default)
# NOTE: per request, the debug configuration toggle has been removed from the
# visible sidebar to simplify UX. If you need to inspect the config while
# developing, enable the DEBUG checkbox here or run in a development branch.



def run_pipeline_on_df(sub_df: pd.DataFrame, max_reviews: int = 0, hf_client=None, provider="mock", ollama_model="llama3.2:3b", adapter=None):
    """Run the processing pipeline and return results dict.

    Results: { df1, sent, summ, trust }
    
    Args:
        provider: "mock", "huggingface", or "ollama"
        ollama_model: model name when using Ollama provider
        adapter: custom adapter configuration (optional)
    """
    if int(max_reviews) > 0 and int(max_reviews) < len(sub_df):
        sub_df = sub_df.head(int(max_reviews)).reset_index(drop=True)
        
    # Use global active_adapter if none provided
    if adapter is None:
        adapter = active_adapter # Use the personalized one from sidebar

    # choose HF client based on settings unless caller provided one
    if hf_client is None:
        hf_client = MockHFClient()
        if provider == "huggingface" and hf_api_key:
            hf_client = HFClient(api_key=hf_api_key)
        elif provider == "huggingface":
            # if HF selected and no explicit key, try HFClient with env var
            try:
                hf_client = HFClient()
            except Exception:
                hf_client = MockHFClient()

    # Pass the provider to detect_fake_reviews so it can use Ollama if selected
    df1 = detect_fake_reviews(
        sub_df, 
        hf_client=hf_client, 
        mock_mode=mock_mode, 
        provider=provider, 
        ollama_model=ollama_model
    )
    
    # For sentiment analysis, provider parameter controls behavior
    sent = analyze_sentiments(df1, hf_client=hf_client, provider=provider, ollama_model=ollama_model)
    df2 = sent["df"]
    
    genuine = df2[df2["is_fake"] == False]
    summ = summarize_reviews(genuine, hf_client=hf_client, mock_mode=mock_mode, provider=provider, ollama_model=ollama_model)
    
    # Compute Trust Score using the CUSTOM ADAPTER (Personalized Weights)
    summary = compute_trust_score(df2, adapter)
    return {"df1": df1, "sent": sent, "summ": summ, "trust": summary}


def _show_representative_reviews(sent_df: pd.DataFrame, label: str = "Representative reviews"):
    """Show one positive and one negative review from the provided sentiment df."""
    if sent_df is None or len(sent_df) == 0:
        st.write("No reviews available to show.")
        return
    # prefer genuine reviews if available
    if "is_fake" in sent_df.columns:
        genuine = sent_df[~sent_df["is_fake"]].copy()
    else:
        genuine = sent_df.copy()

    if genuine is None or len(genuine) == 0:
        genuine = sent_df.copy()

    # If configured, prefer review-level representative selection (reflects majority)
    if rep_mode.startswith("Review"):
        try:
            # filter genuine reviews and exclude mixed/weak ones
            good = genuine.copy()
            if "sentiment_label" in good.columns:
                good = good[~good["sentiment_label"].isin(["mixed"])].copy()
            # positive/negative candidates: split by compound sign
            pos_cands = good[(good.get("compound", 0.0) > 0)]
            neg_cands = good[(good.get("compound", 0.0) < 0)]
            chosen_pos = None
            chosen_neg = None
            if len(pos_cands) > 0:
                median_val = float(pos_cands["compound"].median())
                chosen_pos = pos_cands.iloc[(pos_cands["compound"].sub(median_val).abs()).argsort().iloc[0]]
            if len(neg_cands) > 0:
                median_val_n = float(neg_cands["compound"].median())
                chosen_neg = neg_cands.iloc[(neg_cands["compound"].sub(median_val_n).abs()).argsort().iloc[0]]
            
            if chosen_pos is not None or chosen_neg is not None:
                st.subheader(label)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Positive (representative)**")
                    if chosen_pos is not None:
                        st.write(chosen_pos["reviewText"])
                        meta = {"compound": float(chosen_pos.get("compound", 0.0)), "strength": float(chosen_pos.get("strength", 0.0)), "hf_fake_score": float(chosen_pos.get("hf_fake_score", 0.0))}
                        if "rating" in chosen_pos.index:
                            meta["rating"] = float(chosen_pos.get("rating", None))
                        try:
                            import pandas as _pd
                            st.table(_pd.DataFrame([meta]))
                        except Exception:
                            st.write(meta)
                    else:
                        st.write("No sufficiently positive representative found.")
                with c2:
                    st.markdown("**Negative (representative)**")
                    if chosen_neg is not None:
                        st.write(chosen_neg["reviewText"])
                        meta2 = {"compound": float(chosen_neg.get("compound", 0.0)), "strength": float(chosen_neg.get("strength", 0.0)), "hf_fake_score": float(chosen_neg.get("hf_fake_score", 0.0))}
                        if "rating" in chosen_neg.index:
                            meta2["rating"] = float(chosen_neg.get("rating", None))
                        try:
                            import pandas as _pd
                            st.table(_pd.DataFrame([meta2]))
                        except Exception:
                            st.write(meta2)
                    else:
                        st.write("No sufficiently negative representative found.")
                return
        except Exception:
            pass

    # Fallback to simple sorting
    if "compound" in genuine.columns:
        pos = genuine.sort_values("compound", ascending=False).iloc[0]
        neg = genuine.sort_values("compound", ascending=True).iloc[0]
    else:
        pos = genuine.iloc[0]
        neg = genuine.iloc[-1]

    st.subheader(label)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Positive example**")
        st.write(pos["reviewText"])
        meta = {"compound": float(pos.get("compound", 0.0)), "strength": float(pos.get("strength", 0.0))}
        try:
            import pandas as _pd
            st.table(_pd.DataFrame([meta]))
        except Exception:
            st.write(meta)
    with col2:
        st.markdown("**Negative example**")
        st.write(neg["reviewText"])
        meta2 = {"compound": float(neg.get("compound", 0.0)), "strength": float(neg.get("strength", 0.0))}
        try:
            import pandas as _pd
            st.table(_pd.DataFrame([meta2]))
        except Exception:
            st.write(meta2)


if tab == "Demo sample":
    st.info("Using built-in phone sample dataset for demo.")
    df = pd.read_csv("data/phone_sample_reviews.csv")
    try:
        df = ensure_review_columns(df)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    st.subheader("Demo: sample reviews")
    # Show a compact demo preview limited to columns up to 'price' to keep the UI clean
    demo_cols = _cols_up_to_price(df)
    try:
        st.dataframe(df[demo_cols].head(10).reset_index(drop=True))
    except Exception:
        st.dataframe(df.head(10).reset_index(drop=True))

    # Correct call to run_pipeline_on_df passing the active_adapter
    # For demo, we use default adapter unless user overrides in main UI (which is not shown for demo)
    default_adapter = get_adapter()
    out = run_pipeline_on_df(df, provider=selected_provider, ollama_model=ollama_model, adapter=default_adapter)
    trust = out["trust"]
    st.markdown("## Demo Final Trust Score")
    try:
        st.metric("Final Trust Score", f"{trust['final_score']:.1f}")
    except Exception:
        st.write("Final Trust Score:", trust.get("final_score"))
    st.write("Confidence:", trust.get("confidence"))
    st.markdown("---")

    st.subheader("Sentiment summary")
    sent_summary = out.get("sent", {}).get("summary") or {}
    if sent_summary:
        # show basic counts and a short human-readable summary if available
        st.markdown("**Sentiment summary (counts):**")
        try:
            import pandas as _pd
            st.table(_pd.DataFrame([sent_summary]))
        except Exception:
            st.write(sent_summary)
    else:
        st.info("No sentiment summary available.")

    st.subheader("Aspect breakdown & summaries")
    asp_break = out.get("summ", {}).get("aspect_breakdown") or {}
    if asp_break:
        for asp, txt in asp_break.items():
            with st.expander(f"{asp.replace('_',' ').title()}"):
                st.write(txt)
    else:
        st.info("No aspect breakdown available.")


elif tab == "Upload CSV":
    uploaded = st.file_uploader("Upload reviews CSV (must have reviewText or similar)", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV with reviews. The app will try to detect product/model column automatically.")
    else:
        if "uploaded_filename" not in st.session_state or st.session_state.get("uploaded_filename") != uploaded.name:
            st.session_state["uploaded_df"] = pd.read_csv(uploaded)
            st.session_state["uploaded_filename"] = uploaded.name
        df = st.session_state.get("uploaded_df")

        try:
            df = ensure_review_columns(df)
        except ValueError as err:
            st.error(str(err) + " — please upload a CSV with a review text column.")
            st.stop()

        df, product_col = ensure_product_column(df)
        # Do not show the raw CSV immediately. The app will display
        # a dynamic table of the genuine reviews (those NOT detected as
        # fake) only after the user clicks "Analyze selection". This
        # keeps the upload step lightweight and moves the processed
        # preview into the pipeline stage where it is meaningful.
        st.info("CSV uploaded. Configure options below and click 'Analyze selection' to preview the processed (genuine) reviews used for analysis.")
        # Clear any previous analysis state when a new file is uploaded
        for k in ["last_results", "last_df_rows", "last_genuine_df", "last_analysis_adapter_weights", "_genuine_accumulator_df"]:
            if k in st.session_state:
                del st.session_state[k]
        st.markdown("---")
        st.info("Step 1: (optional) choose price range, product grouping and number of rows (n) to analyze.")
        
        # Price detection and filtering logic
        df, price_col = detect_price_column(df)
        if price_col:
            vals = df[price_col].dropna().astype(float)
            pmin = float(vals.min()) if not vals.empty else 0.0
            pmax = float(vals.max()) if not vals.empty else 1500.0
            pmin = max(0.0, min(1500.0, pmin))
            pmax = max(0.0, min(1500.0, pmax))
            col1, col2 = st.columns(2)
            with col1:
                min_price = st.number_input("Min Price (USD)", min_value=0.0, max_value=1500.0, value=pmin, step=1.0, format="%.2f")
            with col2:
                max_price = st.number_input("Max Price (USD)", min_value=0.0, max_value=1500.0, value=pmax, step=1.0, format="%.2f")
            
            if min_price > max_price:
                st.error("Min Price must be <= Max Price")
                st.stop()
            
            try:
                df = df[df[price_col].notna() & (df[price_col].astype(float) >= float(min_price)) & (df[price_col].astype(float) <= float(max_price))].reset_index(drop=True)
            except Exception:
                df[price_col] = df[price_col].map(lambda x: _parse_price(x))
                df = df[df[price_col].notna() & (df[price_col] >= float(min_price)) & (df[price_col] <= float(max_price))].reset_index(drop=True)
            if df.empty:
                st.warning("No phones found in the selected price range. Adjust the range or upload a different dataset.")
                st.stop()

        # Brand detection and filtering
        if product_col is not None:
            df, brand_list = detect_brands(df, product_col)
            brand_choice = st.selectbox("Filter by brand", ["All brands"] + brand_list)
            if brand_choice and brand_choice != "All brands":
                df = df[df["brand"].fillna("").str.lower() == brand_choice.lower()].reset_index(drop=True)
                if df.empty:
                    st.warning("No phones found for selected brand in this selection.")
                    st.stop()

        max_reviews = st.number_input("Max reviews to analyze per selection (0 = all)", min_value=0, value=0, step=1)

        # Product grouping
        products = []
        if product_col is not None:
            products = df[product_col].fillna("(unknown)").astype(str).unique().tolist()
            grouping_map = group_models(products)
            display_products = ["All (per-product)"] + list(grouping_map.keys())
            if len(display_products) > 1:
                choice = st.selectbox("Select product group (or 'All (per-product)')", display_products)
            else:
                choice = display_products[0]
        else:
            grouping_map = {}
            choice = None

        # === DYNAMIC PERSONALIZATION UI (MAIN AREA) ===
        st.markdown("---")
        st.subheader("🎯 Personalize Your Analysis")
        st.markdown("Rank your top priorities. The system will use this to weight the scores.")

        default_adapter = get_adapter()
        all_aspects = list(default_adapter["aspect_weights"].keys())
        # weights pool from high to low
        available_weights = sorted(default_adapter["aspect_weights"].values(), reverse=True)

        # Use multiselect for ordering
        user_priorities = st.multiselect(
            "Select priorities (ordered by importance):",
            options=all_aspects,
            default=None,
            help="First item = 18%, Second = 16%, etc. Unselected items get remaining low weights."
        )
        
        # Display current priorities horizontally like tags
        if user_priorities:
            st.markdown("**Your Priority Ranking:**")
            priority_html = ""
            for i, p in enumerate(user_priorities):
                if i < len(available_weights):
                    weight_pct = available_weights[i] * 100
                    priority_html += f"<span style='background-color: #e6f3ff; color: #0068c9; padding: 4px 8px; border-radius: 4px; margin-right: 8px; font-weight: 500;'>{i+1}. {p.title()} ({weight_pct:.0f}%)</span>"
                else:
                    # Should not happen if we have enough weights, but safe guard
                    priority_html += f"<span style='background-color: #f0f2f6; padding: 4px 8px; border-radius: 4px; margin-right: 8px;'>{i+1}. {p.title()}</span>"
            
            st.markdown(priority_html, unsafe_allow_html=True)
            
            # Add a "Reset" button functionality (Streamlit doesn't have a direct reset for multiselect, 
            # but we can clear the list in the UI by just deselecting)
            # st.caption("To reset, clear the selection box.")
            
            # Build custom adapter
            final_order = user_priorities + [a for a in all_aspects if a not in user_priorities]
            new_weights_map = dict(zip(final_order, available_weights))
            active_adapter = default_adapter.copy()
            active_adapter["aspect_weights"] = new_weights_map
            
            # Show the implicit/default priorities for the rest
            if len(user_priorities) < len(all_aspects):
                remaining = [a for a in all_aspects if a not in user_priorities]
                remaining_html = "<span style='color:gray; font-size:0.9em;'>Remaining (auto-assigned): "
                start_idx = len(user_priorities)
                for i, p in enumerate(remaining):
                    w_idx = start_idx + i
                    if w_idx < len(available_weights):
                        w_pct = available_weights[w_idx] * 100
                        remaining_html += f"{p.title()} ({w_pct:.0f}%), "
                remaining_html = remaining_html.rstrip(", ") + "</span>"
                st.markdown(remaining_html, unsafe_allow_html=True)

        else:
            st.info("Using default balanced priorities. Select aspects above to customize.")
            active_adapter = default_adapter
        
        st.markdown("---")
        # ==============================================

        analyze_btn = st.button("Analyze selection")

        # ensure hf_client exists
        # If fast_local_mode is enabled, force MockHFClient to avoid remote calls
        if fast_local_mode:
            hf_client = MockHFClient()
            used_provider = "mock"
        else:
            hf_client = MockHFClient()
            if selected_provider == "huggingface" and hf_api_key:
                hf_client = HFClient(api_key=hf_api_key)
                used_provider = "huggingface"
            elif selected_provider == "huggingface":
                try:
                    hf_client = HFClient()
                    used_provider = "huggingface"
                except Exception:
                    hf_client = MockHFClient()
                    used_provider = "mock"
            else:
                used_provider = selected_provider

        if analyze_btn:
            results = {}
            # Progress UI: create a progress bar and status placeholders
            # determine which product groups to analyze based on user choice
            if choice is None:
                selected_bases = list(grouping_map.keys())
            elif choice == "All (per-product)":
                selected_bases = list(grouping_map.keys())
            else:
                # only analyze the chosen product group
                selected_bases = [choice]

            total_products = len(selected_bases)
            # Each product runs through 4 logical steps: fake-detect, sentiment, summarize, trust
            steps_per_product = 4
            total_steps = max(1, total_products * steps_per_product)
            progress_bar = st.progress(0)
            status_text = st.empty()
            step_text = st.empty()

            # Process each group
            for idx, base in enumerate(selected_bases):
                variants = grouping_map.get(base, [])
                # update status for which product we're starting
                status_text.info(f"Analyzing product {idx+1}/{total_products}: '{base}'")
                # compute base step offset for this product
                base_step = idx * steps_per_product
                step_counter = 0
                # refresh progress for starting this product
                progress_bar.progress(int(((base_step + step_counter) / total_steps) * 100))
                sub = df[df[product_col].fillna("(unknown)").astype(str).isin(variants)].reset_index(drop=True)
                if len(sub) == 0:
                    continue
                if int(max_reviews) > 0:
                    sub = sub.head(int(max_reviews)).reset_index(drop=True)
                # --- MANUAL PIPELINE STEPS FOR PER-PRODUCT ANALYSIS ---
                # Step A: Fake detection (Passing provider explicitly)
                step_text.markdown("**Step A:** Fake detection")
                df1 = detect_fake_reviews(
                    sub, 
                    hf_client=hf_client, 
                    mock_mode=mock_mode, 
                    provider=selected_provider, 
                    ollama_model=ollama_model
                )
                # update progress after fake detection
                step_counter += 1
                progress_bar.progress(int(((base_step + step_counter) / total_steps) * 100))
                fake_count = int(df1.get("is_fake", pd.Series([False] * len(df1))).sum())
                n_total = len(df1)
                fake_pct = 100.0 * fake_count / n_total if n_total > 0 else 0.0
                # Step B: Sentiment Analysis
                step_text.markdown("**Step B:** Sentiment analysis")
                sent_res = analyze_sentiments(df1, hf_client=hf_client, provider=selected_provider, ollama_model=ollama_model)
                # update progress after sentiment
                step_counter += 1
                progress_bar.progress(int(((base_step + step_counter) / total_steps) * 100))
                sent_df = sent_res["df"]

                # Step C: Summarize
                step_text.markdown("**Step C:** Summarization")
                genuine = sent_df[~sent_df["is_fake"]].reset_index(drop=True) if "is_fake" in sent_df.columns else sent_df
                summ = summarize_reviews(genuine, hf_client=hf_client, mock_mode=mock_mode, provider=selected_provider, ollama_model=ollama_model)
                # update progress after summarization
                step_counter += 1
                progress_bar.progress(int(((base_step + step_counter) / total_steps) * 100))

                # Step D: Trust score (USING PERSONALIZED ADAPTER)
                step_text.markdown("**Step D:** Compute trust score")
                trust = compute_trust_score(sent_df, active_adapter)
                # update progress after trust scoring
                step_counter += 1
                progress_bar.progress(int(((base_step + step_counter) / total_steps) * 100))

                # Store the genuine reviews used for summarization/LLM steps so
                # the UI can display precisely what was passed on to the models.
                results[base] = {
                    "df1": df1,
                    "sent": sent_res,
                    "genuine_df": genuine,
                    "summ": summ,
                    "trust": trust,
                    "fake_count": fake_count,
                    "fake_pct": fake_pct,
                    "n_total": n_total,
                }

                # accumulate across products into a single DataFrame for display
                existing = st.session_state.get("_genuine_accumulator_df")
                if existing is None:
                    st.session_state["_genuine_accumulator_df"] = genuine.copy()
                else:
                    st.session_state["_genuine_accumulator_df"] = pd.concat([existing, genuine], ignore_index=True)

            # finalize progress
            progress_bar.progress(100)
            status_text.success("Analysis complete")
            step_text.empty()

            # Create summary rows
            rows = [
                {"product": p, "n_reviews": r.get("n_total"), "fake_count": r.get("fake_count"), "fake_pct": f"{r.get('fake_pct'):.1f}%", "final_score": r["trust"].get("final_score"), "confidence": r["trust"].get("confidence")} 
                for p, r in results.items()
            ]
            df_rows = pd.DataFrame(rows)
            st.session_state['last_results'] = results
            st.session_state['last_df_rows'] = df_rows
            st.session_state['last_analysis_provider'] = selected_provider
            st.session_state['last_analysis_mock_mode'] = mock_mode
            st.session_state['last_analysis_ollama_model'] = ollama_model if selected_provider == "ollama" else None
            # Store used adapter weights for display reference
            st.session_state['last_analysis_adapter_weights'] = active_adapter["aspect_weights"]

        # Display results logic
        if st.session_state.get('last_results'):
            results = st.session_state['last_results']
            df_rows = st.session_state.get('last_df_rows', pd.DataFrame())
            used_weights = st.session_state.get('last_analysis_adapter_weights', default_adapter["aspect_weights"])
            
            # Show active settings
            analysis_provider = st.session_state.get('last_analysis_provider', 'unknown')
            analysis_mock_mode = st.session_state.get('last_analysis_mock_mode', True)
            analysis_ollama_model = st.session_state.get('last_analysis_ollama_model')
            
            provider_display = {
                "mock": "Mock (fast/deterministic)",
                "huggingface": "Hugging Face (remote API)",
                "ollama": f"Ollama ({analysis_ollama_model})" if analysis_ollama_model else "Ollama (local LLM)"
            }
            
            st.info(f"📊 **Analysis Results** | Sentiment Provider: **{provider_display.get(analysis_provider, analysis_provider)}** | Fake Detection: **{'Mock Mode' if analysis_mock_mode else 'Real Mode'}**")

            # Show the dynamic table of genuine reviews that were actually
            # used for summarization/LLM steps. This appears after analysis
            # and reflects the rows that passed fake-detection.
            with st.expander("Genuine reviews used for analysis", expanded=False):
                cum = st.session_state.get("_genuine_accumulator_df")
                if cum is not None and not cum.empty:
                    st.markdown("**All genuine rows (combined across selected products):**")
                    cols = _cols_up_to_price(cum)
                    try:
                        st.dataframe(cum[cols].reset_index(drop=True))
                    except Exception:
                        st.dataframe(cum.reset_index(drop=True))
                else:
                    st.info("No genuine reviews available yet. Run 'Analyze selection' to populate this view.")

                # Per-product breakouts
                for prod_name, rdata in results.items():
                    with st.expander(f"{prod_name} — Genuine reviews", expanded=False):
                        gdf = rdata.get('genuine_df')
                        if gdf is None or gdf.empty:
                            st.write("No genuine reviews for this product (all reviews flagged as fake or none passed filters).")
                        else:
                            cols = _cols_up_to_price(gdf)
                            try:
                                st.dataframe(gdf[cols].reset_index(drop=True))
                            except Exception:
                                st.dataframe(gdf.reset_index(drop=True))

            if choice == "All (per-product)":
                st.markdown("**Sort results (interactive)**")
                sort_field = st.selectbox("Sort by", ["final_score", "n_reviews", "fake_count", "fake_pct", "confidence"], index=0, key="sort_field")
                sort_dir = st.radio("Order", ["Descending", "Ascending"], index=0, horizontal=True, key="sort_dir")
                asc = sort_dir == "Ascending"

                tmp = df_rows.copy()
                if sort_field == "fake_pct":
                    tmp["_fake_pct_num"] = tmp["fake_pct"].str.replace("%", "").astype(float)
                    df_sorted = tmp.sort_values(by=["_fake_pct_num"], ascending=asc)
                else:
                    df_sorted = tmp.sort_values(by=[sort_field], ascending=asc, na_position='last')

                st.table(df_sorted.reset_index(drop=True))

                # Interactive table actions
                scores = [(r_name, r_data.get("trust", {}).get("final_score")) for r_name, r_data in results.items()]
                scores = [(n, (s if s is not None else float("-inf"))) for n, s in scores]
                if scores:
                    max_prod, max_score = max(scores, key=lambda t: (t[1] is not None, t[1]))
                    min_prod, min_score = min(scores, key=lambda t: (t[1] is not None, t[1]))
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Max score:** {max_score if max_score not in (None, float('-inf')) else 'N/A'} — {max_prod}")
                        if st.button(f"View max", key="view_max_button"):
                            st.session_state['pinned_product'] = max_prod
                            st.session_state['selected_product'] = max_prod
                    with col2:
                        st.markdown(f"**Min score:** {min_score if min_score not in (None, float('-inf')) else 'N/A'} — {min_prod}")
                        if st.button(f"View min", key="view_min_button"):
                            st.session_state['pinned_product'] = min_prod
                            st.session_state['selected_product'] = min_prod

                # Expanders for details
                pinned = st.session_state.get('pinned_product')
                ordered = []
                if pinned and pinned in results:
                    ordered.append(pinned)
                for prod in results.keys():
                    if prod not in ordered:
                        ordered.append(prod)

                for prod in ordered:
                    r = results.get(prod)
                    expanded = prod == st.session_state.get('selected_product') or prod == st.session_state.get('pinned_product')
                    with st.expander(f"Details: {prod}", expanded=bool(expanded)):
                        st.markdown(f"**Fake reviews:** {r.get('fake_count')} of {r.get('n_total')} ({r.get('fake_pct'):.1f}%)")
                        # Compact trust summary instead of raw JSON — show only Final score
                        tr = r.get('trust', {}) or {}
                        colm, colv = st.columns([1,2])
                        with colm:
                            st.markdown("**Final score**")
                        with colv:
                            try:
                                st.markdown(f"{tr.get('final_score', 'N/A')}")
                            except Exception:
                                st.write(tr)
                        st.subheader("Balanced summary")
                        st.write(r.get("summ", {}).get("balanced_summary"))
                        
                        # Insert nested "Detailed analysis" expander containing
                        # one sub-expander per adapter aspect (battery, camera, ...)
                        with st.expander("Detailed analysis", expanded=False):
                            # aspect list in preferred display order
                            detail_aspects = ["battery", "camera", "performance", "display", "build", "software", "connectivity", "value"]
                            # try to get available data from the result
                            asp_scores = (r.get("trust", {}) or {}).get("aspect_scores", {}) or {}
                            asp_counts = (r.get("trust", {}) or {}).get("aspect_counts", {}) or {}
                            asp_texts = (r.get("summ", {}) or {}).get("aspect_breakdown", {}) or {}
                            sent_df = (r.get('sent', {}) or {}).get('df')
                            # keywords and min mentions from adapter for sentence matching
                            try:
                                from src.adapters.phone_adapter import ASPECT_KEYWORDS, MIN_MENTIONS_FOR_CONFIDENCE
                            except Exception:
                                ASPECT_KEYWORDS = {}
                                MIN_MENTIONS_FOR_CONFIDENCE = {}

                            for asp in detail_aspects:
                                score_val = asp_scores.get(asp)
                                header = f"{asp.replace('_', ' ').title()}"
                                if score_val is not None:
                                    header = f"{header} — Score: {float(score_val):.1f}"
                                # each aspect gets its own nested expander
                                with st.expander(header, expanded=False):
                                    # show aspect-level text summary if available
                                    txt = asp_texts.get(asp)
                                    if txt:
                                        st.write(txt)

                                    # show mention count (if present)
                                    mention_ct = asp_counts.get(asp)
                                    if mention_ct is not None:
                                        st.markdown(f"**Mentions:** {mention_ct}")

                                    # show up to 3 example sentences from genuine reviews that mention an aspect keyword
                                    kws = ASPECT_KEYWORDS.get(asp, [])
                                    if sent_df is not None and 'reviewText' in sent_df.columns and kws:
                                        examples = []
                                        for txt_row in sent_df['reviewText'].fillna("").astype(str).tolist():
                                            lowered = txt_row.lower()
                                            for k in kws:
                                                if k in lowered:
                                                    # split into sentences and pick ones containing keyword
                                                    for sent in [s.strip() for s in __import__('re').split(r'[.!?]+', txt_row) if s.strip()]:
                                                        if k in sent.lower():
                                                            examples.append(sent.strip())
                                                            break
                                            if len(examples) >= 3:
                                                break
                                        if examples:
                                            st.markdown("**Example sentences (up to 3):**")
                                            for ex in examples[:3]:
                                                st.write(f"- {ex}")

                        aspect_scores = r.get("trust", {}).get("aspect_scores", {}) or {}
                        available_aspects = {k: v for k, v in aspect_scores.items() if v is not None}
                        best_aspect = None
                        if available_aspects:
                            try:
                                best_aspect = max(available_aspects.items(), key=lambda kv: float(kv[1]))[0]
                            except Exception:
                                best_aspect = next(iter(available_aspects.keys()))
                            st.markdown(f"**Key aspect (by score):** {best_aspect} — {available_aspects.get(best_aspect):.2f}")
            else:
                # Single product detailed view
                cached_results = st.session_state.get('last_results', {})
                if choice and choice in cached_results:
                    r = cached_results[choice]
                    st.markdown(f"### Product Details: {choice}")
                    trust = r.get("trust", {})
                    try:
                        st.metric("Final Trust Score", f"{trust.get('final_score', 0.0):.1f}")
                    except Exception:
                        st.write(f"Final Trust Score: {trust.get('final_score', 0.0):.1f}")

                    # Show fake-review counts and percentage (these reviews were excluded from LLM analysis)
                    fake_count = r.get('fake_count', None)
                    n_total = r.get('n_total', None)
                    if fake_count is not None and n_total is not None:
                        try:
                            fake_pct = (100.0 * fake_count / n_total) if n_total > 0 else 0.0
                            col_a, col_b = st.columns([1,2])
                            with col_a:
                                st.metric("Fake reviews", f"{fake_count}/{n_total}")
                            with col_b:
                                st.metric("Excluded from analysis", f"{fake_pct:.1f}%")
                            st.caption("Note: Detected fake reviews (shown above) were excluded from the LLM sentiment/summarization steps and do not contribute to aspect scores.")
                        except Exception:
                            # Fallback simple display
                            st.write(f"Fake reviews: {fake_count} of {n_total}")

                    # Show the genuine reviews that were used for LLM/summarization
                    with st.expander("Genuine reviews used for analysis", expanded=True):
                        gdf = r.get('genuine_df')
                        if gdf is None or gdf.empty:
                            st.info("No genuine reviews available for this product (all reviews may have been flagged as fake or filtered out).")
                        else:
                            cols = _cols_up_to_price(gdf)
                            try:
                                st.dataframe(gdf[cols].reset_index(drop=True))
                            except Exception:
                                st.dataframe(gdf.reset_index(drop=True))
                    
                    st.subheader("Aspect Quality Scores")
                    aspect_scores = trust.get("aspect_scores", {}) or {}
                    available_aspects = {k: v for k, v in aspect_scores.items() if v is not None}
                    
                    if available_aspects:
                        cols = st.columns(2)
                        for idx, (aspect, score) in enumerate(sorted(available_aspects.items(), key=lambda x: x[1], reverse=True)):
                            col_idx = idx % 2
                            with cols[col_idx]:
                                try:
                                    st.metric(aspect.replace('_', ' ').title(), f"{score:.1f}")
                                except Exception:
                                    st.write(f"**{aspect.replace('_', ' ').title()}:** {score:.1f}")
                    else:
                        st.info("No aspect scores available for this product.")
                    # --- Score breakdown expander (weights, aspect scores, contributions) ---
                    st.markdown("")
                    with st.expander("Score breakdown — weights & contributions", expanded=False):
                        st.markdown("**What you see here:** each aspect's configured weight (percentage), the computed aspect score (0-100), and its weighted contribution to the final trust score.")
                        st.markdown("---")
                        # Show both the user's configured weights (personalization) and the
                        # effective normalized weights used in the final calculation.
                        weights_effective = trust.get('aspect_weights_effective', {}) or {}
                        contributions = trust.get('aspect_contributions', {}) or {}
                        configured = used_weights if 'used_weights' in locals() else st.session_state.get('last_analysis_adapter_weights', {})
                        # Build a compact table with configured weight, effective weight, score and contribution
                        # Order aspects dynamically by the user's configured priority (highest configured weight first)
                        all_asps = ["battery", "camera", "performance", "display", "build", "software", "connectivity", "value"]
                        # ensure configured weights are numeric
                        cfg_map = {a: float(configured.get(a, 0.0)) for a in all_asps}
                        # order by configured weight descending so the user's priorities appear first
                        ordered_asps = sorted(all_asps, key=lambda a: cfg_map.get(a, 0.0), reverse=True)
                        rows = []
                        for asp in ordered_asps:
                            cfg_w = cfg_map.get(asp, 0.0)
                            eff_w = float(weights_effective.get(asp, 0.0))
                            score = aspect_scores.get(asp)
                            # compute contribution from effective weight and score (score may be None)
                            contrib_val = 0.0
                            if score is not None:
                                try:
                                    contrib_val = float(score) * eff_w
                                except Exception:
                                    contrib_val = 0.0
                            rows.append({
                                "aspect": asp.replace('_',' ').title(),
                                "configured_weight": f"{cfg_w*100:.0f}%",
                                "effective_weight": f"{eff_w*100:.0f}%",
                                "score": f"{score:.1f}" if score is not None else "N/A",
                                "contribution": f"{contrib_val:.2f}",
                                "calculation": (f"{score:.1f} * {eff_w:.2f} = {contrib_val:.2f}" if score is not None else "-")
                            })
                        try:
                            import pandas as _pd
                            st.table(_pd.DataFrame(rows))
                        except Exception:
                            for r in rows:
                                st.markdown(f"- **{r['aspect']}** — Weight: {r['weight_pct']} | Score: {r['score']} | Contribution: {r['contribution']}")

                    st.subheader("Detailed Aspect Breakdown")
                    aspect_breakdown = r.get("summ", {}).get("aspect_breakdown", {}) or {}
                    
                    # Methodology explanation
                    with st.expander("ℹ️ What is the Detailed Aspect Breakdown?", expanded=False):
                        from src.adapters.phone_adapter import ASPECT_KEYWORDS, MIN_MENTIONS_FOR_CONFIDENCE
                        st.markdown("### Component Scoring & Coverage (Personalized)")
                        aspect_counts_local = trust.get('aspect_counts', {}) or {}
                        for aspect in ["battery", "camera", "performance", "display", "build", "software", "connectivity", "value"]:
                            aspect_title = aspect.replace('_', ' ').title()
                            # Use the personalized weights from session state
                            weight = used_weights.get(aspect, 0.0)
                            min_req = MIN_MENTIONS_FOR_CONFIDENCE.get(aspect, 3)
                            kws = ASPECT_KEYWORDS.get(aspect, [])
                            score_val = aspect_scores.get(aspect)
                            mention_ct = aspect_counts_local.get(aspect, 0)
                            status = (f"✅ Scored: {score_val:.1f}" if score_val is not None else f"⚠️ Not scored ({mention_ct}/{min_req} mentions)")
                            line = f"**{aspect_title}** | Weight: {weight:.0%} | Mentions: {mention_ct} (min {min_req}) | {status}\nKeywords: " + ", ".join(kws[:8]) + (" …" if len(kws) > 8 else "")
                            st.markdown(line)
                            
                    # Display aspect details
                    if aspect_breakdown and available_aspects:
                        for aspect in sorted(available_aspects.keys(), key=lambda k: available_aspects[k], reverse=True):
                            aspect_text = aspect_breakdown.get(aspect)
                            if aspect_text:
                                header = f"{aspect.replace('_', ' ').title()} - Score: {available_aspects.get(aspect, 0):.1f}"
                                with st.expander(header):
                                    st.write(aspect_text)
                    # Also show aspects that were NOT included (excluded) with reason and examples
                    excluded = [a for a, v in (trust.get('aspect_scores', {}) or {}).items() if v is None]
                    if excluded:
                        st.subheader("Excluded aspects (not scored)")
                        # gather genuine review texts to show small examples
                        sent_df = r.get('sent', {}).get('df', None)
                        try:
                            import pandas as _pd
                        except Exception:
                            _pd = None

                        if sent_df is not None and _pd is not None and 'reviewText' in sent_df.columns:
                            genuine_texts_df = sent_df[~sent_df.get('is_fake', _pd.Series([False]*len(sent_df)))] if 'is_fake' in sent_df.columns else sent_df
                        else:
                            genuine_texts_df = None

                        from src.adapters.phone_adapter import ASPECT_KEYWORDS, MIN_MENTIONS_FOR_CONFIDENCE
                        for aspect in excluded:
                            mention_ct = (trust.get('aspect_counts', {}) or {}).get(aspect, 0)
                            min_req = MIN_MENTIONS_FOR_CONFIDENCE.get(aspect, 3)
                            kws = ASPECT_KEYWORDS.get(aspect, [])
                            with st.expander(f"{aspect.replace('_',' ').title()} — Not scored ({mention_ct}/{min_req} mentions)"):
                                st.markdown(f"**Reason:** Not enough mentions (required {min_req}, found {mention_ct}).")
                                # Keywords removed by user request — only show reason & examples
                                # show up to 3 example sentences from genuine reviews that mention any keyword
                                if genuine_texts_df is not None:
                                    examples = []
                                    for txt in genuine_texts_df['reviewText'].fillna("").astype(str).tolist():
                                        lowered = txt.lower()
                                        for k in kws:
                                            if k in lowered:
                                                # split into sentences and pick ones containing keyword
                                                for sent in [s.strip() for s in __import__('re').split(r'[.!?]+', txt) if s.strip()]:
                                                    if k in sent.lower():
                                                        examples.append(sent.strip())
                                                        break
                                        if len(examples) >= 3:
                                            break
                                    if examples:
                                        st.markdown("**Example mentions (up to 3):**")
                                        for ex in examples[:3]:
                                            st.write(f"- {ex}")
                                    else:
                                        st.markdown("No example sentences found in genuine reviews for this aspect.")
                                else:
                                    st.markdown("No genuine review texts available to show examples.")
                else:
                    st.warning("Please run 'All (per-product)' analysis first.")
                    if st.button("Analyze this product now"):
                        # Single analysis fallback
                        pass # (Implementation omitted for brevity, user likely won't hit this path with per-product selected)

elif tab == "Analyze YouTube Video":
    st.info("Analyze a video's transcript and fuse with a placeholder text score (mock).")
    vid = st.text_input("YouTube URL or ID", "")
    if st.button("Analyze video") and vid.strip():
        res = analyze_video(vid.strip(), mock_mode=mock_mode)
        if res.get("error"):
            st.error("Transcript not available for this video.")
        else:
            st.write(res)
            text_score = st.slider("Placeholder text review trust score (0-100)", 0, 100, 75)
            fused = fuse_scores(float(text_score), float(res.get("video_score", 0.0)))
            st.metric("Video Score", f"{res.get('video_score', 0.0):.1f}")
            st.metric("Fused Final Score", f"{fused:.1f}")