import pandas as pd
import re


df = pd.read_csv(
    "data/metadata.csv"
)


def tokenize(sentence):

    tokens = re.findall(
        r"\w+(?:'\w+)?|[^\w\s]", #the goal is to find all the words and punctuation marks in the sentence.
        sentence.lower()
    )

    return tokens


sentence = df.iloc[0]["sentence"]

tokens = tokenize(sentence)

print("Original sentence:")
print(sentence)

print("\nTokens:")
print(tokens)