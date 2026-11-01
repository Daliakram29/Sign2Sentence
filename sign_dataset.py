import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset


class Sign2SentenceDataset(Dataset):

    def __init__(self, metadata_path):
        self.df = pd.read_csv(metadata_path)

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

        return sequence, sentence


# Test the dataset
dataset = Sign2SentenceDataset(
    "data/metadata.csv"
)

sequence, sentence = dataset[0]

print("Number of samples:", len(dataset))
print("Sequence shape:", sequence.shape)
print("Sequence type:", type(sequence))
print("Sentence:", sentence)