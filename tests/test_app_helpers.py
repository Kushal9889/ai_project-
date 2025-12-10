import pandas as pd
from src.app_helpers import ensure_review_columns, ensure_product_column


def test_ensure_review_columns_basic():
    df = pd.DataFrame({"text": ["Good phone", "Bad phone"], "Rating": ["5", "2"]})
    out = ensure_review_columns(df)
    assert "reviewText" in out.columns
    assert "rating" in out.columns
    # rating should be numeric when possible
    assert out["rating"].dtype.kind in "fi"
    assert out["reviewText"].iloc[0] == "Good phone"


def test_ensure_product_column_detect():
    df = pd.DataFrame({"model": ["X1", "X2"], "review": ["ok", "bad"]})
    df2, prod_col = ensure_product_column(df)
    assert prod_col == "product"
    assert "product" in df2.columns
    assert df2["product"].iloc[0] in ("X1", "X2")
import pandas as pd
from src.app_helpers import ensure_review_columns, ensure_product_column


def test_ensure_review_columns_basic():
    df = pd.DataFrame({"text": ["Good phone", "Bad phone"], "Rating": ["5", "2"]})
    out = ensure_review_columns(df)
    assert "reviewText" in out.columns
    assert "rating" in out.columns
    # rating should be numeric when possible
    assert out["rating"].dtype.kind in "fi"
    assert out["reviewText"].iloc[0] == "Good phone"


def test_ensure_product_column_detect():
    df = pd.DataFrame({"model": ["X1", "X2"], "review": ["ok", "bad"]})
    df2, prod_col = ensure_product_column(df)
    assert prod_col == "product"
    assert "product" in df2.columns
    assert df2["product"].iloc[0] in ("X1", "X2")
