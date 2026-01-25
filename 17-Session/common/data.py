import pandas as pd

def load_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "customer_id": [101, 102, 101, 103],
        "month": ["Jan", "Feb", "Mar", "Mar"],
        "sales_usd": [120.5, 300.0, 220.3, 150.0]
    })
