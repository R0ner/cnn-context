"""Evaluation metric for BugNIST kaggle challenge.
Requirements: numpy, pandas, scipy
"""

# %%
import re

import numpy as np
import pandas as pd
import scipy
from scipy import optimize
from scipy.spatial.distance import cdist


class ParticipantVisibleError(Exception):
    """
    Custom exception class for errors that should be shown to participants.
    """

    pass


def match_precision_recall(matches: np.ndarray, pred_labels: np.ndarray, true_labels: np.ndarray, eps: float = 1e-6) -> tuple:
    """
    Calculate precision and recall for detection and class matching.

    Args:
        matches: (2, K) array of box to center matches
        pred_labels: (N,) array of predicted labels
        true_labels: (M,) array of true labels
        eps: Small value to avoid division by zero

    Returns:
        Tuple containing precision and recall for detection and class matching
    """
    pred_match_labels = pred_labels[matches[0]]
    true_match_labels = true_labels[matches[1]]
    matches_class = matches[:, pred_match_labels == true_match_labels]

    # Detection metrics
    precision_detect = matches.shape[1] / pred_labels.shape[0]
    recall_detect = matches.shape[1] / true_labels.shape[0]
    f1_detect = 2 * precision_detect * recall_detect / (precision_detect + recall_detect + eps)

    # Class metrics
    precision_classes = matches_class.shape[1] / pred_labels.shape[0]
    recall_classes = matches_class.shape[1] / true_labels.shape[0]
    f1_classes = 2 * precision_classes * recall_classes / (precision_classes + recall_classes + eps)

    return f1_detect, precision_detect, recall_detect, f1_classes, precision_classes, recall_classes


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = "filename") -> float:
    """
    Calculate the score for the given solution and submission dataframes.

    Args:
        solution (pd.DataFrame): DataFrame containing the ground truth data.
        submission (pd.DataFrame): DataFrame containing the submission data.
        row_id_column_name (str): Name of the column containing row IDs.

    Returns:
        float: Score for the submission.
    """
    f1_scores = []
    for i in range(len(submission)):
        if pd.isna(submission.iloc[i, 1]):
            raise ParticipantVisibleError(f"Nan value not supported. Found on line {i} in submission file")

        # Extracts data for each line
        pred_centerpoints = submission.iloc[i, 1].replace(" ", "").rstrip(";").split(";")  # Removes whitespace and last ';' if applicable
        sol_centerpoints = solution.iloc[i, 1].split(";")
        pred_filename = str(submission.iloc[i, 0])
        sol_filename = str(solution.iloc[i, 0])

        if pred_filename != sol_filename:
            raise ParticipantVisibleError("Internal error: solution and submission are not lined up")

        if len(pred_centerpoints) % 4 != 0:
            raise ParticipantVisibleError(
                f"Submission for file {pred_filename}, index {i} could not be separated based on ';' into segmentations of size 4. Instead, got a list of size {len(pred_centerpoints)} % 4 != 0"
            )

        # Extract center coordinates and labels from submission and solution
        filtered_centerpoints = [float(item) for item in pred_centerpoints if not re.search("[a-zA-Z]", item)]
        filtered_true_centerpoints = [float(item) for item in sol_centerpoints if not re.search("[a-zA-Z]", item)]
        pred_centers = np.array(list(zip(filtered_centerpoints[::3], filtered_centerpoints[1::3], filtered_centerpoints[2::3])))
        true_centers = np.array(list(zip(filtered_true_centerpoints[::3], filtered_true_centerpoints[1::3], filtered_true_centerpoints[2::3])))

        # Converts labels to numbers
        index_to_label = np.array(["sl", "bc", "ma", "gh", "ac", "bp", "bf", "cf", "bl", "ml", "wo", "pp"])
        label_to_index = {k.lower(): i for i, k in enumerate(index_to_label)}
        label_to_index["gp"] = label_to_index["bp"]

        try:
            pred_labels = np.array([label_to_index[label.lower()] for label in pred_centerpoints[::4]])
            true_labels = np.array([label_to_index[label.lower()] for label in sol_centerpoints[::4]])
        except KeyError:
            raise ParticipantVisibleError(f"Invalid class label in {pred_centerpoints[::4]}, should match any of {index_to_label}")

        # Calculate cost matrix and perform matching
        cost = cdist(pred_centers, true_centers)
        matches = np.array(scipy.optimize.linear_sum_assignment(cost), dtype=np.int32)

        # Filter matches based on matched labels
        matched_box_labels = pred_labels[matches[0]]
        matched_center_labels = true_labels[matches[1]]
        matches = matches[:, matched_box_labels == matched_center_labels]

        # Calculate F1 score
        f1_score = match_precision_recall(matches, pred_labels, true_labels)[3]
        f1_scores.append(f1_score)

    return np.mean(f1_scores)


# %%
def main(pred, target):
    # loads data from csv files
    df_pred = pd.read_csv(pred)
    df_target = pd.read_csv(target)

    # calculates score
    f1_mean = score(df_pred, df_target, "filename")

    # prints score
    print(f"F1 score: {f1_mean}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pred", type=str, default="Sample_solution.csv", help="Path to csv file containing predictions (default: %(default)s)")
    parser.add_argument("-t", "--target", type=str, default="validation.csv", help="Path to csv file containing solution file (default: %(default)s)")

    args = parser.parse_args()
    print("# Args options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)
    print("")
    main(args.target, args.pred)
# %%
