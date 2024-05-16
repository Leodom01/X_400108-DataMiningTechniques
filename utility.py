"""
utility functions for splitting, training, and evaluation
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from lightgbm import LGBMRanker
from tqdm import tqdm
import re


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


def preprocess_data(train_set, test_set, test_label=False):
    # lightgbm does not like weird characters in column names
    train_set = train_set.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    test_set = test_set.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    # Train data
    X_train = train_set.loc[:, ~train_set.columns.isin(['srch_id','target_label', 'position', 'gross_booking_usd'])]
    y_train = train_set['target_label']
    qid_train = train_set['srch_id']

    # Test data
    X_test = test_set.loc[:, ~test_set.columns.isin(['srch_id','target_label', 'position', 'gross_booking_usd'])]
    if test_label:
        y_test = test_set['target_label']
    else:
        y_test = None
    qid_test = test_set['srch_id']

    return X_train, y_train, qid_train, X_test, y_test, qid_test


def train_lgbm_ranker(X_train, y_train, train_groups, params=None):
    if params is None:
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "n_estimators": 2000,
            "learning_rate": 0.05
        }

    gbm = LGBMRanker(**params)
    gbm.fit(X_train, y_train, group=train_groups)

    return gbm


def predict_and_generate_submission(model, X_test, q_id_test, output_file):
    """
    Predicts and generates a submission file based on the model's predictions.
    :param model: The trained model
    :param X_test: Test data
    :param q_id_test: Query IDs for the test data
    :param output_file: Output file path for the submission
    """
    predictions = []
    # Use tqdm to track progress over unique groups
    for group in tqdm(np.unique(q_id_test), desc='Processing groups'):
        preds = model.predict(X_test[q_id_test == group])
        predictions.extend(preds)

    X_test['preds'] = predictions
    X_test['srch_id'] = q_id_test

    result = X_test.sort_values(by=['srch_id', 'preds'], ascending=[True, False])
    result[['srch_id', 'prop_id']].reset_index(drop=True).to_csv(output_file, index=False)


from sklearn.metrics import ndcg_score
from tqdm import tqdm

def calculate_ndcg(model, X_test, y_test, q_id_test, k=5):
    """
    Calculate NDCG score for the given model and test data.
    :param model: Trained model
    :param X_test: Test data
    :param y_test: Ground truth labels for the test data
    :param q_id_test: Query IDs for the test data
    :param k: Number of top predictions to consider for NDCG calculation
    :return: List of NDCG scores
    """
    ndcg_scores = []
    qids = np.unique(q_id_test)
    for qid in tqdm(qids, desc='Calculating NDCG scores'):
        y = y_test[q_id_test == qid].values.flatten()
        p = model.predict(X_test[q_id_test == qid])
        ndcg_scores.append(ndcg_score([y], [p], k=k))
    return np.mean(ndcg_scores)




