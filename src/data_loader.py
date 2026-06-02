import os
import pandas as pd


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found"
        )

    return pd.read_csv(path)
