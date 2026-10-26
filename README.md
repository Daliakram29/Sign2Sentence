## Sign2Sentence

Sign2Sentence is my graduation project exploring real-time translation of continuous American Sign Language (ASL) into natural English text and AI-generated speech.

The goal is to build a practical communication tool rather than only classify individual signs. The system is intended to process a person signing through a webcam or video, analyze the movement across time, generate a clear English sentence, and later convert the translated sentence into speech.

Project Idea

The planned pipeline is:

Video / Webcam
      ↓
Video frame processing
      ↓
Hand + Pose + Face landmarks
      ↓
Deep learning sequence model
      ↓
Natural English sentence
      ↓
AI-generated speech

The first version will focus on ASL → English.

## Future versions may explore:

real-time continuous translation

AI-generated speech using ElevenLabs

expression-aware speech generation

Arabic sign-language support

integration with online meetings such as Zoom or Google Meet

tools for creating translated and narrated social-media videos

## Current Stage

The project is currently in the data preparation and computer-vision learning stage.

Current work includes:

learning OpenCV video processing

working with video frames and webcam input

preparing the sign-language dataset

exploring landmark extraction using MediaPipe

continuing deep-learning study using Deep Learning for Coders with fastai & PyTorch

## Dataset

The project currently uses the How2Sign dataset.

How2Sign is a multimodal dataset containing continuous American Sign Language videos paired with English translations.

For the initial development stage, I am using:

Green Screen RGB Clips — Frontal View

Validation split

Manually re-aligned English translations

Only a small subset of the dataset will initially be used while developing and testing the preprocessing pipeline.

Dataset website:

https://how2sign.github.io/



## Planned Technologies

- Python
- OpenCV
- MediaPipe
- NumPy
- PyTorch
- fastai
- LSTM / GRU
- Transformers
- ElevenLabs
- Streamlit or another web interface

The final technology stack may change as the project develops.



## Project Status

🚧 Work in progress

This repository will be updated throughout the development of my graduation project as I learn, experiment, train models, and build the final application.
