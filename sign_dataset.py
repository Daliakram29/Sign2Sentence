import pandas as pd
import numpy as np
import torch
import re
from collections import Counter
from torch.utils.data import Dataset

def tokenize(sentence):

    return re.findall(
        r"\w+(?:'\w+)?|[^\w\s]",
        sentence.lower()
    )

class Sign2SentenceDataset(Dataset):

    def __init__(self, metadata_path):
        self.df = pd.read_csv(metadata_path)
        token_counts = Counter()

        for sentence in self.df["sentence"]:
            tokens = tokenize(sentence)
            token_counts.update(tokens)

        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

        for token in token_counts:

            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # Load the processed landmark sequence
        sequence = np.load(row["sequence_file"])

        # Convert NumPy array → PyTorch tensor
        sequence = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        # English target sentence
        sentence = row["sentence"]

        tokens = tokenize(sentence)

        token_ids = [
            self.vocab["<BOS>"]
        ]

        for token in tokens:
            token_ids.append(
                self.vocab.get(
                    token,
                    self.vocab["<UNK>"]
                )
            )

        token_ids.append(
            self.vocab["<EOS>"]
        )

        token_tensor = torch.tensor(
            token_ids,
            dtype=torch.long
        )

        return sequence, token_tensor, sentence


# Test the dataset
if __name__ == "__main__":

    dataset = Sign2SentenceDataset(
        "data/metadata.csv"
    )

    sequence, token_tensor, sentence = dataset[0]

    print("Number of samples:", len(dataset))

    print("\nVideo shape:")
    print(sequence.shape)

    print("\nToken tensor:")
    print(token_tensor)

    print("\nToken shape:")
    print(token_tensor.shape)

    print("\nOriginal sentence:")
    print(sentence)

    print("\nVocabulary size:")
    print(len(dataset.vocab))