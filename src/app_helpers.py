import pandas as pd


def ensure_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common review/rating column names to 'reviewText' and 'rating'.

    Raises ValueError if no review text column found.
    """
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    if "reviewtext" in lower_map:
        df = df.rename(columns={lower_map["reviewtext"]: "reviewText"})
    else:
        # Added 'comment' to this list to support your specific CSV file
        alternates = ["review_text", "text", "review", "reviews", "comments", "comment", "body", "content", "feedback"]
        for alt in alternates:
            if alt in lower_map:
                df = df.rename(columns={lower_map[alt]: "reviewText"})
                break
    # If we still don't have a reviewText column, try a heuristic: pick the most
    # text-like column (object dtype with long-ish strings and many non-nulls).
    if "reviewText" not in df.columns:
        # candidate columns: object (string) columns
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        best = None
        best_score = 0.0
        for c in obj_cols:
            non_null = df[c].notna().sum()
            if non_null == 0:
                continue
            # compute median length of string values (coerce to str)
            lengths = df[c].dropna().astype(str).map(len)
            median_len = float(lengths.median()) if len(lengths) > 0 else 0.0
            # score favors columns with many non-nulls and longer text
            score = (non_null / max(1, len(df))) * (median_len / 100.0)
            if score > best_score:
                best_score = score
                best = c
        # accept candidate if it looks sufficiently text-like
        if best is not None and (best_score > 0.05 or median_len > 30):
            df = df.rename(columns={best: "reviewText"})
    if "rating" in lower_map:
        df = df.rename(columns={lower_map["rating"]: "rating"})
    else:
        rating_alts = ["ratings", "stars", "ratingvalue", "reviewrating", "score", "star"]
        for ralt in rating_alts:
            if ralt in lower_map:
                df = df.rename(columns={lower_map[ralt]: "rating"})
                break
    # Normalize rating values to numeric (0..5 scale) when present
    if "rating" in df.columns:
        import re

        def _parse_rating(val):
            # pass through numeric types
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            if isinstance(val, (int, float)):
                try:
                    return float(val)
                except Exception:
                    return None
            s = str(val).strip()
            if s == "":
                return None
            # percent -> scale 0..5
            if "%" in s:
                m = re.search(r"(\d+(?:\.\d+)?)%", s)
                if m:
                    try:
                        pct = float(m.group(1))
                        return max(0.0, min(5.0, (pct / 100.0) * 5.0))
                    except Exception:
                        pass
            # fraction patterns like '4/5' or '4.0 out of 5'
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*(\d+(?:\.\d+)?)", s, flags=re.IGNORECASE)
            if m:
                try:
                    num = float(m.group(1))
                    den = float(m.group(2))
                    if den > 0:
                        return max(0.0, min(5.0, (num / den) * 5.0))
                except Exception:
                    pass
            # otherwise extract first numeric token (handles '5.0 out of 5 stars...')
            m_all = re.findall(r"(\d+(?:\.\d+)?)", s)
            if m_all:
                try:
                    return float(m_all[0])
                except Exception:
                    return None
            return None

        df["rating"] = df["rating"].map(_parse_rating)
    if "reviewText" in df.columns:
        return df
    raise ValueError("Input dataframe is missing required column 'reviewText'")


def ensure_product_column(df: pd.DataFrame):
    """Detect and normalize a product/model column to 'product' if present.

    Returns (df, product_col_name_or_None)
    """
    lower_map = {c.lower(): c for c in df.columns}
    candidates = ["product", "product_name", "model", "phone", "title", "item", "name"]
    for cand in candidates:
        if cand in lower_map:
            df = df.rename(columns={lower_map[cand]: "product"})
            return df, "product"
    for c in df.columns:
        if any(k in c.lower() for k in ("model", "product", "phone")):
            df = df.rename(columns={c: "product"})
            return df, "product"
    return df, None