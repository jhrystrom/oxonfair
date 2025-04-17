import argparse
import copy
import time
from itertools import combinations
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equal_opportunity_difference,
)
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from oxonfair import FairPredictor, dataset_loader
from oxonfair import group_metrics as gm


class FairnessMetrics(TypedDict):
    equal_opportunity: float
    min_recall: float
    accuracy: float
    precision: float
    recall: float


def calculate_metrics(
    test_groups: np.ndarray,
    test_labels: np.ndarray,
    predictions: list[str] | np.ndarray,
) -> FairnessMetrics:
    metric_frame = MetricFrame(
        metrics={"recall": recall_score},
        y_true=test_labels,
        y_pred=predictions,
        sensitive_features=test_groups,
    )
    recall_by_group = {
        f"recall_group_{key}": value
        for key, value in metric_frame.by_group.to_dict()["recall"].items()
    }
    return {
        "min_recall": metric_frame.group_min()["recall"],
        "equal_opportunity": equal_opportunity_difference(
            y_true=test_labels,
            y_pred=predictions,
            sensitive_features=test_groups,
        ),
        "accuracy": accuracy_score(y_true=test_labels, y_pred=predictions),
        "precision": precision_score(y_true=test_labels, y_pred=predictions),
        "recall": recall_score(y_true=test_labels, y_pred=predictions),
        "demographic_parity": demographic_parity_difference(
            y_true=test_labels, y_pred=predictions, sensitive_features=test_groups
        ),
        **recall_by_group,
    }


def calculate_disagreement_rates(ensemble_preds):
    """
    Calculate the disagreement rate for all pairs of predictions in an ensemble.

    Args:
        ensemble_preds: A list of lists where each sublist contains the predictions
                        for one instance across all predictors.

    Returns:
        A numpy array containing the disagreement rate for each pair of predictors.
    """
    # Convert to numpy array for easier manipulation
    preds = np.array(ensemble_preds).T  # Transpose to get predictors as rows

    n_predictors = preds.shape[0]
    n_pairs = n_predictors * (n_predictors - 1) // 2  # Number of unique pairs

    # Initialize array to store disagreement rates
    disagreement_rates = np.zeros(n_pairs)

    # Calculate disagreement rate for each pair of predictors
    for pair_idx, (i, j) in enumerate(combinations(range(n_predictors), 2)):
        # Count instances where predictions differ
        disagreements = np.sum(preds[i] != preds[j])
        # Calculate disagreement rate
        disagreement_rates[pair_idx] = disagreements / preds.shape[1]

    return disagreement_rates


def calculate_error_rate_avg(ensemble_preds, y_true):
    """Calculate average error rate across ensemble members"""
    num_members = len(ensemble_preds[0])
    total_error = 0
    for i in range(num_members):
        member_preds = [pred[i] for pred in ensemble_preds]
        member_error = 1 - accuracy_score(y_true=y_true, y_pred=member_preds)
        total_error += member_error
    return total_error / num_members


def calculate_der(preds, y_true):
    """Calculate Disagreement to Error Rate ratio"""
    disagreement = calculate_disagreement_rates(preds).mean()
    average_error = calculate_error_rate_avg(ensemble_preds=preds, y_true=y_true)
    # Avoid division by zero
    if average_error == 0:
        return 0
    return disagreement / average_error


def calculate_der_groups(preds, y_true, groups, is_recall=False):
    """Calculate DER for each group"""
    group_mask = np.array(groups)
    unique_groups = np.unique(group_mask)
    ders = {group: 0 for group in unique_groups}

    for group in unique_groups:
        mask_variable = group_mask == group
        if is_recall:
            logger.debug("Only calculating positive (recall)")
            mask_variable = mask_variable & np.array(y_true)

        y_true_group = np.array(y_true)[mask_variable == 1]
        if len(y_true_group) == 0:
            ders[group] = 0
            continue

        group_indices = np.where(mask_variable)[0]
        group_preds = [pred for i, pred in enumerate(preds) if i in group_indices]

        if len(group_preds) == 0:
            ders[group] = 0
            continue

        assert (
            len(group_preds) == y_true_group.shape[0]
        ), f"{len(group_preds)=} != {y_true_group.shape[0]=}"
        group_der = calculate_der(preds=group_preds, y_true=y_true_group)
        ders[group] = group_der

    return ders


def aggregate_scores(scores: list[np.ndarray]) -> list[bool]:
    final_preds = consolidate_predictions(scores)
    return majority_vote(final_preds)


def aggregate_probas(probas: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    combined = np.vstack(probas).T
    num_members = combined.shape[1]
    return (combined > threshold).sum(axis=1) > num_members / 2


def consolidate_predictions(scores):
    num_preds = len(scores[0])
    # Convert to list[(pred0, pred0, pred0), (pred1, ...]
    final_preds = []
    for pred_index in range(num_preds):
        pred_list = []
        for score_list in scores:
            score = score_list[pred_index]
            pred_list.append(score)
        final_preds.append(pred_list)
    return final_preds


def majority_vote(lists):
    """Perform majority voting across ensemble predictions"""
    return [sum(sublist) > len(sublist) / 2 for sublist in lists]


def train_ensemble(
    train_data,
    val_data,
    test_data,
    num_members=3,
    metric="demographic_parity",
    threshold=0.02,
):
    """Train an ensemble of fair classifiers and evaluate on test data"""

    # Prepare outputs
    ensemble_metrics = []
    individual_metrics = []
    ensemble_models = []

    # Record start time
    start_time = time.perf_counter()

    # Create folds for training ensemble members
    combined_data = {
        "data": np.vstack([train_data["data"], val_data["data"]]),
        "target": np.concatenate([train_data["target"], val_data["target"]]),
        "groups": np.concatenate([train_data["groups"], val_data["groups"]]),
    }

    gss = StratifiedKFold(n_splits=num_members, random_state=42, shuffle=True)
    splits = list(gss.split(combined_data["data"], combined_data["target"]))

    # Train ensemble members on different folds
    for member, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Training ensemble member {member+1}/{num_members}")

        # Create fold-specific train and validation sets
        fold_train_data = {
            "data": combined_data["data"][train_idx],
            "target": combined_data["target"][train_idx],
            "groups": combined_data["groups"][train_idx],
        }

        fold_val_data = {
            "data": combined_data["data"][val_idx],
            "target": combined_data["target"][val_idx],
            "groups": combined_data["groups"][val_idx],
        }

        # Train the base model (XGBoost)
        base_model = xgb.XGBClassifier(random_state=42 + member)
        base_model.fit(X=fold_train_data["data"], y=fold_train_data["target"])

        # Apply fairness constraints
        fair_predictor = FairPredictor(base_model, fold_val_data)
        if metric == "demographic_parity":
            fair_predictor.fit(gm.accuracy, gm.demographic_parity, threshold)
        elif metric == "equal_opportunity":
            fair_predictor.fit(gm.accuracy, gm.equal_opportunity, threshold)
        elif metric == "min_recall":
            fair_predictor.fit(gm.accuracy, gm.recall.min, threshold)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Save the model
        ensemble_models.append(fair_predictor)

        # Evaluate on validation set
        val_predictions_data = {
            "data": fold_val_data["data"],
            "target": fold_val_data[
                "target"
            ],  # Include target even though it's not used for prediction
            "groups": fold_val_data["groups"],
        }

        val_preds = fair_predictor.predict(val_predictions_data)
        val_metrics = calculate_metrics(
            test_groups=fold_val_data["groups"],
            test_labels=fold_val_data["target"],
            predictions=val_preds,
        )

        fair_predictor.evaluate_fairness()
        fairness = fair_predictor.evaluate_fairness(
            metrics=gm.default_fairness_measures | {"min_recall": gm.recall.min}
        ).assign(classifier=member, metric_type="fairness")

        val_metrics["member"] = member
        val_metrics["dataset"] = "validation"
        individual_metrics.append(val_metrics)

        # Plot fairness frontier
        plt.figure(figsize=(10, 6))
        fair_predictor.plot_frontier()
        plt.title(f"Fairness Frontier - Member {member+1}")
        plt.savefig(f"fairness_frontier_member_{member+1}-{metric}.png")
        plt.close()

    end_time = time.perf_counter()
    training_time = end_time - start_time

    # Evaluate ensemble on test set
    logger.info("Evaluating ensemble on test set")

    # Get predictions from all ensemble members
    ensemble_preds = []
    test_predictions_data = {"data": test_data["data"], "target": test_data["target"]}
    for model in ensemble_models:
        preds = model.predict(test_predictions_data)
        ensemble_preds.append(preds)

    # Calculate ensemble metrics
    final_preds = aggregate_scores(ensemble_preds)
    formatted_preds = consolidate_predictions(ensemble_preds)

    ensemble_test_metrics = calculate_metrics(
        test_groups=test_data["groups"],
        test_labels=test_data["target"],
        predictions=final_preds,
    )

    individual_test_metrics = fair_predictor.evaluate_fairness(
        data=test_data,
        metrics=gm.default_fairness_measures | {"min_recall": gm.recall.min},
    )

    ensemble_test_metrics["model_type"] = "ensemble"
    ensemble_test_metrics["num_members"] = num_members
    ensemble_test_metrics["training_time"] = training_time

    # Calculate DER (Disagreement to Error Rate)
    ensemble_der = calculate_der(preds=formatted_preds, y_true=test_data["target"])
    ensemble_test_metrics["der"] = ensemble_der

    # Calculate group-specific DER
    group_ders = calculate_der_groups(
        preds=formatted_preds,
        y_true=test_data["target"],
        groups=test_data["groups"],
        is_recall=True,
    )

    for group, der_value in group_ders.items():
        ensemble_test_metrics[f"der_group_{group}"] = der_value

    # Evaluate individual models on test set for comparison
    for member, model in enumerate(ensemble_models):
        indiv_preds = model.predict(test_predictions_data)
        indiv_metrics = calculate_metrics(
            test_groups=test_data["groups"],
            test_labels=test_data["target"],
            predictions=indiv_preds,
        )
        indiv_metrics["member"] = member
        indiv_metrics["dataset"] = "test"
        indiv_metrics["model_type"] = "individual"
        individual_metrics.append(indiv_metrics)

    return ensemble_test_metrics, individual_metrics, ensemble_models


def merge_ethnicity_groups(data, groups_to_merge, merge_name=" Other"):
    """Merge specified ethnicity groups into a single category"""
    data_copy = copy.deepcopy(data)
    for group in groups_to_merge:
        data_copy["groups"][data_copy["groups"] == group] = merge_name
    return data_copy


def main(
    metric: str = "equal_opportunity",
    members: int = 3,
    threshold: float = 0.02,
    iterations: int = 1,
):
    # Set up output directory
    output_dir = Path("./ensemble_results")
    output_dir.mkdir(exist_ok=True)

    # Load adult dataset with ethnicity as sensitive attribute
    logger.info("Loading Adult dataset with ethnicity as sensitive attribute")

    # Analyze original ethnicity distribution
    train_data, val_data, test_data = dataset_loader.adult(
        "race",
        train_proportion=0.6,
        test_proportion=0.2,  # Remaining 0.2 goes to validation
        seperate_groups=True,
        seed=42,
    )
    ethnicity_dist = pd.Series(train_data["groups"]).value_counts()
    logger.info(f"Original ethnicity distribution: {ethnicity_dist}")

    # Define different groupings to test
    groupings = [
        {"name": "original", "description": "Original 5 ethnicity groups", "merge": []},
        {
            "name": "4groups",
            "description": "4 ethnicity groups (merging smallest)",
            "merge": [ethnicity_dist.index[-1]],  # Merge the smallest group
        },
        {
            "name": "3groups",
            "description": "3 ethnicity groups",
            "merge": [
                ethnicity_dist.index[-1],
                ethnicity_dist.index[-2],
            ],  # Merge the two smallest groups
        },
        {
            "name": "2groups",
            "description": "2 ethnicity groups (White vs. Others)",
            "merge": [group for group in ethnicity_dist.index if group != " White"],
        },
    ]

    # Create results dataframe
    results = []

    # Train ensemble for each grouping
    for grouping in groupings:
        if grouping["name"] != "original":
            continue
        logger.info(
            f"Training with grouping: {grouping['name']} - {grouping['description']}"
        )
        for iteration in tqdm(range(iterations), desc="Iterations", unit="iteration"):
            train_data, val_data, test_data = dataset_loader.adult(
                "race",
                train_proportion=0.6,
                test_proportion=0.2,  # Remaining 0.2 goes to validation
                seperate_groups=True,
                seed=42,
            )

            # Merge groups if needed
            if grouping["merge"]:
                train_data_merged = merge_ethnicity_groups(
                    train_data, grouping["merge"]
                )
                val_data_merged = merge_ethnicity_groups(val_data, grouping["merge"])
                test_data_merged = merge_ethnicity_groups(test_data, grouping["merge"])

                # Log the new group distribution
                merged_dist = pd.Series(train_data_merged["groups"]).value_counts()
                logger.info(f"Merged ethnicity distribution: {merged_dist}")
            else:
                train_data_merged = train_data
                val_data_merged = val_data
                test_data_merged = test_data

            # Train ensemble
            ensemble_metrics, individual_metrics, ensemble_models = train_ensemble(
                train_data_merged,
                val_data_merged,
                test_data_merged,
                num_members=members,
                metric=metric,
                threshold=threshold,
            )

            # Add grouping info to metrics
            ensemble_metrics["grouping"] = grouping["name"]
            ensemble_metrics["description"] = grouping["description"]
            ensemble_metrics["num_groups"] = len(np.unique(test_data_merged["groups"]))
            ensemble_metrics["iteration"] = iteration

            # Save to results
            results.append(ensemble_metrics)

            # Save individual metrics
            indiv_df = pd.DataFrame(individual_metrics)
            indiv_df["grouping"] = grouping["name"]
            indiv_df["iteration"] = iteration
            indiv_df.to_csv(
                output_dir / f"individual_metrics_{grouping['name']}-i{iteration}.csv",
                index=False,
            )

            # Plot group-specific metrics
            group_metrics = {
                k: v
                for k, v in ensemble_metrics.items()
                if k.startswith("recall_group_")
            }
            if group_metrics:
                plt.figure(figsize=(10, 6))
                plt.bar(group_metrics.keys(), group_metrics.values())
                plt.title(f"Group Recall - {grouping['name']}")
                plt.xticks(rotation=45, ha="right")
                plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(output_dir / f"group_recall_{grouping['name']}.png")
                plt.close()

            test_predictions_data = {
                "data": test_data_merged["data"],
                "target": test_data_merged["target"],
            }
            probas = [
                ensemble_model.predict_proba(test_predictions_data)[:, 1]
                for ensemble_model in ensemble_models
            ]
            prediction_thresholds = np.linspace(0, 1, 100)

            all_metrics = []
            for pred_threshold in prediction_thresholds:
                preds = aggregate_probas(probas, threshold=pred_threshold)
                metrics = calculate_metrics(
                    test_groups=test_data_merged["groups"],
                    test_labels=test_data_merged["target"],
                    predictions=preds,
                )
                metrics["threshold"] = pred_threshold
                metrics["iteration"] = iteration
                all_metrics.append(metrics)

            single_proba = probas[:1]
            single_metrics = []
            for pred_threshold in prediction_thresholds:
                preds = aggregate_probas(single_proba, threshold=pred_threshold)
                metrics = calculate_metrics(
                    test_groups=test_data_merged["groups"],
                    test_labels=test_data_merged["target"],
                    predictions=preds,
                )
                metrics["threshold"] = pred_threshold
                metrics["iteration"] = iteration
                single_metrics.append(metrics)

        pd.DataFrame(single_metrics).to_csv(
            output_dir / f"single_threshold_metrics_{grouping['name']}.csv", index=False
        )

        pd.DataFrame(all_metrics).to_csv(
            output_dir
            / f"threshold_metrics_{grouping['name']}-n{members}-i{iterations}.csv",
            index=False,
        )

    # Save overall results
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        output_dir / f"ensemble_results-n{members}-i{iterations}.csv", index=False
    )

    # Create summary plot
    plt.figure(figsize=(12, 8))

    # Plot accuracy vs fairness
    x = results_df["accuracy"]
    y = results_df[f"{metric}"]
    labels = results_df["grouping"]

    plt.scatter(x, y, s=100)
    for i, label in enumerate(labels):
        plt.annotate(
            label, (x.iloc[i], y.iloc[i]), xytext=(5, 5), textcoords="offset points"
        )

    plt.xlabel("Accuracy")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Accuracy vs {metric.replace('_', ' ').title()} for Different Groupings")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_fairness.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ensemble on Adult dataset with ethnicity groups"
    )
    parser.add_argument(
        "--members", type=int, default=3, help="Number of ensemble members"
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="Number of iterations to run"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["demographic_parity", "equal_opportunity", "min_recall"],
        default="demographic_parity",
        help="Fairness metric to optimize",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.02, help="Fairness threshold"
    )
    args = parser.parse_args()
    main(
        members=args.members,
        metric=args.metric,
        threshold=args.threshold,
        iterations=args.iterations,
    )
