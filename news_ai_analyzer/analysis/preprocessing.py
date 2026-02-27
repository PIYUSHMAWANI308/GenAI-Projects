import pandas as pd
import numpy as np
from genai.llm_explainer import explain_credibility_llm
from genai.headline_generator import generate_neutral_headline


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # feature engineering
    df["text_length"] = df["text"].apply(len)
    df["title_length"] = df["title"].apply(len)

    return df


def dataset_statistics(df: pd.DataFrame) -> dict:
    return {
        "total_articles": len(df),
        "avg_text_length": np.mean(df["text_length"]),
        "max_text_length": np.max(df["text_length"]),
        "min_text_length": np.min(df["text_length"])
    }


def add_llm_explanations(df: pd.DataFrame) -> pd.DataFrame:
    df["credibility_explanation"] = df.apply(
        explain_credibility_llm,
        axis=1
    )
    return df

def add_neutral_headlines(df):
    df["neutral_headline"] = df["title"].apply(generate_neutral_headline)
    return df
