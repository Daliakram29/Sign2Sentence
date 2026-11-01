import pandas as pd
from pathlib import Path

VIDEO_PATH = r"C:\Users\dalia\Downloads\val_rgb_front_clips\raw_videos\3EjKvwck6ss_0-1-rgb_front.mp4"

CSV_PATH = r"C:\Users\dalia\Downloads\how2sign_realigned_val.csv"


df = pd.read_csv(CSV_PATH, sep="\t")


video_name = Path(VIDEO_PATH).stem

row = df[df["SENTENCE_NAME"] == video_name]

if not row.empty:
    sentence = row.iloc[0]["SENTENCE"]

    print("Video:", video_name)
    print("English sentence:", sentence)

else:
    print("No matching sentence found.")