def explain_credibility(row , avg_length) :
    """
    Generate a human-readable explanation for the credibility score
    """

    explanations = []

    # Source-based explanation

    if row["source"] in ["BBC" , "Reuters" , "CNN"] :
        explanations.append(
            f"The article comes from a well-known source ({row['source']}), which increases credibility."
        )
    elif row["source"] == "FakeNewsDaily" :
        explanations.append(
            "The article is from a known unreliable source, which significantly lowers credibility."
        )
    else :
        explanations.append(
            "The article is from an unknown or less-established source, which raises credibility concerns."
        )

    # length-based explanation

    if row["text_length"] < avg_length:
        explanations.append(
        "The article is shorter than the average news length, which may reduce credibility."
    )
    else:
        explanations.append(
        "The article length is close to or above the dataset average, suggesting sufficient detail."
    )

    # Final score explanation
    explanations.append(
        f"Overall, these factors resulted in a credibility score of {row['credibility_score']} out of 100."
    )

    return " ".join(explanations)