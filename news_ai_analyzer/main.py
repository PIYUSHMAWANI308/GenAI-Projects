from analysis.preprocessing import (
    load_and_clean_data,
    dataset_statistics,
    add_llm_explanations,
    add_neutral_headlines
)
from analysis.credibility_score import calculate_credibility


def main():
    df = load_and_clean_data("data/news.csv")
    df = calculate_credibility(df)

    stats = dataset_statistics(df)
    print("Dataset Statistics:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    df = add_llm_explanations(df)
    df = add_neutral_headlines(df)

    print("\nFinal Output:")
    print(
        df[
            ["title", "neutral_headline", "source", "credibility_score"]
        ].head()
    )


if __name__ == "__main__":
    main()