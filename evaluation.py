"""
A bunch of utility functions for evaluation
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd


def train_test_split_by_group(data, id_column, test_size=0.2, random_state=None):
    """
    Split a dataset by groups defined by a specific column.

    Parameters:
    - data: pandas DataFrame, the dataset to be split.
    - id_column: str, the name of the column containing the group IDs.
    - test_size: float, optional (default=0.2), the proportion of the dataset to include in the test split.
    - random_state: int or RandomState instance, optional (default=None), control the randomness of the shuffling.

    Returns:
    - train_set: pandas DataFrame, the training set.
    - test_set: pandas DataFrame, the test set.
    """
    # Create GroupShuffleSplit object
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Group by the specified column and apply GroupShuffleSplit
    groups = data[id_column]
    train_idx, test_idx = next(gss.split(data, groups=groups))

    # Split the dataset into train and test sets
    train_set = data.iloc[train_idx]
    test_set = data.iloc[test_idx]

    return train_set, test_set

def ndcg_at_k(true_relevance, predicted_relevance, k):
    # Ensure lengths of true_relevance and predicted_relevance are equal
    if len(true_relevance) != len(predicted_relevance):
        raise ValueError("Lengths of true_relevance and predicted_relevance must be equal")


    # Get the true relevance scores based on the sorted indices
    true_relevance_sorted = [true_relevance[i] for i in predicted_relevance]

    # Calculate DCG (Discounted Cumulative Gain) at k
    dcg_at_k = 0
    for i in range(min(k, len(true_relevance))):
        dcg_at_k += (2 ** true_relevance_sorted[i] - 1) / np.log2(i + 2)

    # Sort the true relevance scores
    true_relevance_sorted_desc = sorted(true_relevance, reverse=True)

    # Calculate ideal DCG at k
    idcg_at_k = 0
    for i in range(min(k, len(true_relevance))):
        idcg_at_k += (2 ** true_relevance_sorted_desc[i] - 1) / np.log2(i + 2)

    # Calculate NDCG at k
    if idcg_at_k == 0:
        ndcg_at_k = 0
    else:
        ndcg_at_k = dcg_at_k / idcg_at_k

    return ndcg_at_k