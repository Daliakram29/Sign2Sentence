import pandas as pd
import re
from collections import Counter
import torch

df = pd.read_csv("data/metadata.csv")


def tokenize(sentence):

    return re.findall(
        r"\w+(?:'\w+)?|[^\w\s]",
        sentence.lower()
    )


# -------------------------
# BUILD VOCABULARY
# -------------------------

token_counts = Counter()

for sentence in df["sentence"]:

    tokens = tokenize(sentence)

    token_counts.update(tokens)


vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3
}


for token in token_counts:

    if token not in vocab:
        vocab[token] = len(vocab)


print("Vocabulary size:", len(vocab))

print("\nFirst vocabulary items:")

for token, token_id in list(vocab.items())[:20]:
    print(token, "→", token_id)


# -------------------------
# TEST ONE SENTENCE
# -------------------------

sentence = df.iloc[0]["sentence"]

tokens = tokenize(sentence)

token_ids = [
    vocab["<BOS>"]
]

for token in tokens:

    token_id = vocab.get(
        token,
        vocab["<UNK>"]
    )

    token_ids.append(token_id)


token_ids.append(
    vocab["<EOS>"]
)


print("\nOriginal sentence:")
print(sentence)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)

token_tensor = torch.tensor(
    token_ids,
    dtype=torch.long
)

print("\nToken tensor:")
print(token_tensor)

print("\nToken tensor shape:")
print(token_tensor.shape)