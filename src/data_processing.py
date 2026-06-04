from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


def dataset_summary(df):
    print("Shape:", df.shape)
    print(df.info())
    return df.describe()


def missing_values(df):
    return (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )


def get_numeric_columns(df):
    return df.select_dtypes(include="number")


def create_aggregate_features(df):

    customer_features = (
        df.groupby("CustomerId")
        .agg(
            TotalTransactionAmount=("Amount", "sum"),
            AverageTransactionAmount=("Amount", "mean"),
            TransactionCount=("TransactionId", "count"),
            TransactionStd=("Amount", "std")
        )
        .reset_index()
    )
    return customer_features


def extract_time_features(df):

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["TransactionHour"] = (
        df["TransactionStartTime"].dt.hour
    )

    df["TransactionDay"] = (
        df["TransactionStartTime"].dt.day
    )

    df["TransactionMonth"] = (
        df["TransactionStartTime"].dt.month
    )

    df["TransactionYear"] = (
        df["TransactionStartTime"].dt.year
    )

    return df


def build_preprocessor(
    numerical_cols,
    categorical_cols
):

    numeric_pipeline = Pipeline([
        ("imputer",
         SimpleImputer(strategy="median")),
        ("scaler",
         StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer",
         SimpleImputer(strategy="most_frequent")),
        ("encoder",
         OneHotEncoder(
             handle_unknown="ignore"
         ))
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numeric_pipeline,
            numerical_cols
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_cols
        )
    ])

    return preprocessor


def create_rfm_features(df):

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date - x.max()
                ).days
            ),
            Frequency=(
                "TransactionId",
                "count"
            ),
            Monetary=(
                "Amount",
                "sum"
            )
        )
        .reset_index()
    )

    return rfm
