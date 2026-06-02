import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(df, column):
    sns.histplot(df[column], bins=50)
    plt.title(f"Distribution of {column}")
    plt.show()


def plot_heatmap(df):
    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        annot=False
    )

    plt.title("Correlation Heatmap")
    plt.show()


def plot_boxplot(df, column):
    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df[column])

    plt.title(f"Boxplot of {column}")
    plt.show()


def plot_monthly_transactions(df):
    df_copy = df.copy()

    df_copy["TransactionStartTime"] = pd.to_datetime(
        df_copy["TransactionStartTime"]
    )

    df_copy["Month"] = (
        df_copy["TransactionStartTime"]
        .dt.month
    )

    monthly = (
        df_copy.groupby("Month")
        .size()
    )

    monthly.plot(
        figsize=(10, 5)
    )

    plt.title("Transactions per Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Transactions")
    plt.show()


def plot_product_category_distribution(df):
    (
        df["ProductCategory"]
        .value_counts()
        .plot(
            kind="bar",
            figsize=(10, 5)
        )
    )

    plt.title("Product Category Distribution")
    plt.xlabel("Product Category")
    plt.ylabel("Count")
    plt.show()
