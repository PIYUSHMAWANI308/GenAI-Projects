import pandas as pd
import numpy as np

# simple rule-based source credibility

SOURCE_SCORES={
    "BBC" : 90 ,
    "Reuters" : 95 , 
    "CNN" : 85 , 
    "Unknown Blog" : 40 ,
    "FakeNewsDaily" : 20
}

def score_source(source : str) -> int :
    # Assign credibility score based on news source
     
    return SOURCE_SCORES.get(source , 50) # default score

def score_text_length(text_length : int , avg_length : float) -> int :
    # Score credibility based on how close text length is to avg length 

    diff = abs(text_length - avg_length)
    penalty = min(diff / avg_length , 1)

    score = 100 - (penalty * 100)
    return int(score)

def calculate_credibility(df : pd.DataFrame) -> pd.DataFrame :
    #combine multiple signals into a final credibility score
    
    avg_length = np.mean(df["text_length"])

    source_scores = df["source"].apply(score_source)
    length_scores = df["text_length"].apply(
        lambda x: score_text_length (x , avg_length)
    )

    # weighted average (Numpy logic)

    df["credibility_score"] = np.average(
        np.vstack([source_scores , length_scores]),
        axis=0 ,
        weights= [0.6 , 0.4]
    ).astype(int)

    return df
