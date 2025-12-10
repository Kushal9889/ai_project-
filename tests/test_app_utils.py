import pandas as pd
from src import app_utils


def test_parse_price_variants():
    assert app_utils._parse_price("$199") == 199.0
    assert app_utils._parse_price("199 USD") == 199.0
    assert app_utils._parse_price("1,299") == 1299.0
    assert app_utils._parse_price(None) is None
    assert app_utils._parse_price("not a price") is None


def test_detect_price_column_and_cols_up_to_price():
    df = pd.DataFrame({"cost": [199, 299], "name": ["a", "b"]})
    new_df, price_col = app_utils.detect_price_column(df)
    assert price_col == "price"
    assert "price" in new_df.columns
    # cols_up_to_price returns columns up to price
    cols = app_utils.cols_up_to_price(new_df)
    assert "price" in cols


def test_normalize_model_name():
    val = app_utils.normalize_model_name("iPhone (12) 8GB RAM")
    assert "Iphone" in val or "iphone" in val.lower()
import pandas as pd
from src import app_utils


def test_parse_price_variants():
    assert app_utils._parse_price("$199") == 199.0
    assert app_utils._parse_price("199 USD") == 199.0
    assert app_utils._parse_price("1,299") == 1299.0
    assert app_utils._parse_price(None) is None
    assert app_utils._parse_price("not a price") is None


def test_detect_price_column_and_cols_up_to_price():
    df = pd.DataFrame({"cost": [199, 299], "name": ["a", "b"]})
    new_df, price_col = app_utils.detect_price_column(df)
    assert price_col == "price"
    assert "price" in new_df.columns
    # cols_up_to_price returns columns up to price
    cols = app_utils.cols_up_to_price(new_df)
    assert "price" in cols


def test_normalize_model_name():
    val = app_utils.normalize_model_name("iPhone (12) 8GB RAM")
    assert "Iphone" in val or "iphone" in val.lower()
