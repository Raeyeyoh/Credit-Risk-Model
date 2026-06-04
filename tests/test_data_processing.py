from src.data_processing import (
    create_aggregate_features
)

import pandas as pd


# test1
def test_aggregate_columns():

    df = pd.DataFrame({
        "CustomerId": ["C1", "C1"],
        "TransactionId": ["T1", "T2"],
        "Amount": [100, 200]
    })

    result = create_aggregate_features(df)

    assert "TotalTransactionAmount" in result.columns


def test_customer_count():

    df = pd.DataFrame({
        "CustomerId": ["C1", "C1"],
        "TransactionId": ["T1", "T2"],
        "Amount": [100, 200]
    })

    result = create_aggregate_features(df)

    assert len(result) == 1
