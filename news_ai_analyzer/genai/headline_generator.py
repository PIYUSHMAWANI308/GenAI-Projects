import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("api key"))

def generate_neutral_headline(original_headline: str) -> str:
    """
    Generate a neutral, factual version of a news headline using an LLM
    """

    prompt = f"""
Rewrite the following news headline to be neutral, factual, and unbiased.
Do not add new information.
Keep it concise.

Original headline:
"{original_headline}"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You rewrite news headlines to be neutral and factual."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()