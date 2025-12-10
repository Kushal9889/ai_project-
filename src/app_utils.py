import re
import pandas as pd
from typing import Tuple, List, Optional


def _parse_price(x) -> Optional[float]:
	"""Parse common price formats into float. Returns None on failure."""
	try:
		if x is None:
			return None
		s = str(x).strip()
		if s == "":
			return None
		s = re.sub(r"[\$,£€]", "", s)
		s = re.sub(r"(?i)\s*usd\b", "", s)
		s = s.replace(',', '')
		s = s.replace('\u00a0', '')
		s = s.strip()
		if s in ['', '-', '–', '—']:
			return None
		m = re.search(r"(\d+(?:\.\d+)?)", s)
		if not m:
			return None
		return float(m.group(1))
	except Exception:
		return None


def detect_price_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
	"""Try to detect a price column and coerce it to float. Returns (df, price_col_or_None)."""
	lower = {c.lower(): c for c in df.columns}
	candidates = ["price", "cost", "amount", "price_usd", "retail_price", "msrp"]
	for cand in candidates:
		if cand in lower:
			df = df.rename(columns={lower[cand]: "price"})
			df["price"] = df["price"].map(_parse_price)
			return df, "price"
	# numeric heuristic
	num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
	for c in num_cols:
		vals = df[c].dropna()
		if len(vals) == 0:
			continue
		within = ((vals >= 0) & (vals <= 1500)).sum() / len(vals)
		if within > 0.6:
			df = df.rename(columns={c: "price"})
			df["price"] = df["price"].astype(float)
			return df, "price"
	# string heuristic
	str_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
	for c in str_cols:
		sample = df[c].dropna().astype(str).head(200)
		if len(sample) == 0:
			continue
		parsed = sample.map(_parse_price)
		success = parsed.notna().sum() / len(parsed)
		if success > 0.6:
			df = df.rename(columns={c: "price"})
			df["price"] = df["price"].map(_parse_price)
			vals = df["price"].dropna()
			if len(vals) > 0 and ((vals >= 0) & (vals <= 1500)).sum() / len(vals) > 0.5:
				return df, "price"
			df = df.rename(columns={"price": c})
	return df, None


def cols_up_to_price(df: pd.DataFrame) -> List[str]:
	"""Return column list up to and including the detected price column (or all columns if none)."""
	if "price" in df.columns:
		cols = list(df.columns)
		idx = cols.index("price")
		return cols[: idx + 1]
	return list(df.columns)


def normalize_model_name(name: str) -> str:
	"""Small normalization for model names used in tests.

	Examples: strip parentheses, collapse extra whitespace, title-case.
	"""
	if name is None:
		return ""
	s = str(name)
	# remove parenthetical content
	s = re.sub(r"\(.*?\)", "", s)
	s = re.sub(r"[^\w\s-]", " ", s)
	s = re.sub(r"\s+", " ", s).strip()
	return s.title()

