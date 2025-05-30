# RNA 3D Structure Prediction with Transformers

This project predicts the 3D coordinates of RNA molecules using a Transformer-based deep learning model. It combines sequence embeddings, multiple sequence alignments (MSA), and position-based features to generate atomic predictions.

# Model Overview
Architecture: Transformer Encoder with positional encoding

Inputs:

Nucleotide sequences

MSA profiles

Relative position encoding

Output: 3D coordinates for each nucleotide (x, y, z)

# Evaluation

Metrics:

Root Mean Square Deviation (RMSD)

TM-score (structural alignment score)

Validation scores and training loss are logged per epoch and plotted automatically.


# Notes
Mean and std for coordinate normalization are computed from the training set and fixed for inference.

Supports data augmentation through Gaussian noise injection during training.
