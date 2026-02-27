import os
from openai import OpenAI
from openai import RateLimitError

client = OpenAI(api_key=os.getenv("api key"))

def explain_credibility_llm(row):
    try:
        prompt = f"""
Article source: {row['source']}
Article length: {row['text_length']} words
Credibility score: {row['credibility_score']} out of 100

Explain briefly why this score makes sense.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You explain news credibility clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        return response.choices[0].message.content.strip()

    except RateLimitError:
        return (
            f"This article received a credibility score of {row['credibility_score']} "
            f"based on its source reliability and content length. "
            f"Longer, structured articles from known sources score higher."
        )