import torch

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from sign_dataset import Sign2SentenceDataset


dataset = Sign2SentenceDataset(
    "data/metadata.csv"
)


def collate_fn(batch):

    sequences, token_tensors, sentences = zip(*batch)

    # -------------------------
    # VIDEO PADDING
    # -------------------------

    video_lengths = torch.tensor(
        [sequence.shape[0] for sequence in sequences]
    )

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0
    )

    max_video_length = padded_sequences.shape[1]

    video_mask = (
        torch.arange(max_video_length)[None, :]
        < video_lengths[:, None]
    )

    # -------------------------
    # TEXT PADDING
    # -------------------------

    text_lengths = torch.tensor(
        [tokens.shape[0] for tokens in token_tensors]
    )

    padded_tokens = pad_sequence(
        token_tensors,
        batch_first=True,
        padding_value=0
    )

    max_text_length = padded_tokens.shape[1]

    text_mask = (
            torch.arange(max_text_length)[None, :]
            < text_lengths[:, None]
    )

    return (
        padded_sequences,
        video_lengths,
        video_mask,
        padded_tokens,
        text_lengths,
        text_mask,
        sentences
    )


# -------------------------
# DATALOADER
# -------------------------

dataloader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
    collate_fn=collate_fn
)


# Get one batch
(
    sequences,
    video_lengths,
    video_mask,
    tokens,
    text_lengths,
    text_mask,
    sentences
) = next(iter(dataloader))

decoder_input = tokens[:, :-1]
decoder_target = tokens[:, 1:]

print("\nDecoder input:")
print(decoder_input)

print("\nDecoder target:")
print(decoder_target)

print("\nDecoder input shape:")
print(decoder_input.shape)

print("\nDecoder target shape:")
print(decoder_target.shape)

# -------------------------
# TEST OUTPUT
# -------------------------

print("Video batch shape:")
print(sequences.shape)

print("\nVideo lengths:")
print(video_lengths)

print("\nVideo mask shape:")
print(video_mask.shape)

print("\nText batch shape:")
print(tokens.shape)

print("\nText lengths:")
print(text_lengths)

print("\nToken batch:")
print(tokens)

print("\nText mask shape:")
print(text_mask.shape)

print("\nText mask for sentence 2:")
print(text_mask[1])