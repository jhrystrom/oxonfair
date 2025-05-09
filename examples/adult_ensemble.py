import argparse
import copy
import random
import time
from itertools import combinations
from pathlib import Path
from typing import TypedDict

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equal_opportunity_difference,
)
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import FixedThresholdClassifier, StratifiedKFold
from tqdm import tqdm

from oxonfair import FairPredictor, dataset_loader
from oxonfair import group_metrics as gm

datasets = {
    "adult": dataset_loader.adult,
    "compas": dataset_loader.compas,
}


class FairnessMetrics(TypedDict):
    equal_opportunity: float
    accuracy: float
    min_recall: float
    precision: float
    recall: float


Model = (
    xgb.XGBClassifier | lgb.LGBMClassifier | CatBoostClassifier | RandomForestClassifier
)


def make_random_model(rng: random.Random) -> Model:
    choice = rng.choice(["xgb", "lgbm", "cat", "rf"])

    if choice == "xgb":
        return xgb.XGBClassifier(
            max_depth=rng.choice([4, 6]),
            subsample=rng.choice([0.6, 0.8, 1.0]),
            random_state=rng.randint(0, 9999),
        )
    if choice == "lgbm":
        return lgb.LGBMClassifier(
            num_leaves=rng.choice([31, 63]),
            feature_fraction=rng.choice([0.6, 0.8, 1.0]),
            random_state=rng.randint(0, 9999),
        )
    if choice == "cat":
        return CatBoostClassifier(
            depth=rng.choice([4, 6]),
            rsm=rng.choice([0.6, 0.8, 1.0]),
            verbose=False,
            random_state=rng.randint(0, 9999),
        )
    return RandomForestClassifier(
        max_depth=rng.choice([None, 10]),
        max_features=rng.choice([0.5, 0.7, 1.0]),
        random_state=rng.randint(0, 9999),
    )


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
    iteration_number: int = 0,
    metric="demographic_parity",
    threshold=0.02,
):
    """Train an ensemble of fair classifiers and evaluate on test data"""
    if num_members % 2 == 0:
        raise ValueError("Number of members must be odd to avoid ties")

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

    val_predictions = []
    val_labels = []
    val_groups = []
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
        base_model = make_random_model(random.Random(42 + member + iteration_number))
        model_name = base_model.__class__.__name__
        logger.debug(f"Base model: {model_name}")
        logger.debug("Ftting base model")
        base_model.fit(X=fold_train_data["data"], y=fold_train_data["target"])
        logger.debug("Base model fitted")

        # Save the model
        ensemble_models.append(base_model)

        # Evaluate on validation set
        val_predictions_data = {
            "data": fold_val_data["data"],
            "target": fold_val_data[
                "target"
            ],  # Include target even though it's not used for prediction
            "groups": fold_val_data["groups"],
        }

        val_preds = base_model.predict_proba(val_predictions_data["data"])
        val_labels.append(fold_val_data["target"])
        val_predictions.append(val_preds)
        val_groups.append(fold_val_data["groups"])
        val_metrics = calculate_metrics(
            test_groups=fold_val_data["groups"],
            test_labels=fold_val_data["target"],
            predictions=base_model.predict(val_predictions_data["data"]),
        )

        val_metrics["member"] = member
        val_metrics["model_name"] = model_name
        val_metrics["split"] = "validation"
        individual_metrics.append(val_metrics)

    predictions = np.concatenate(val_predictions)

    combined_fair_predictor = FairPredictor(
        predictor=None,
        validation_data={
            "data": predictions,
            "target": np.concatenate(val_labels),
            "groups": np.concatenate(val_groups),
        },
    )

    if metric == "min_recall":
        combined_fair_predictor.fit(gm.accuracy, gm.recall.min, threshold)
    elif metric == "demographic_parity":
        combined_fair_predictor.fit(gm.accuracy, gm.demographic_parity, threshold)
    elif metric == "equal_opportunity":
        combined_fair_predictor.fit(gm.accuracy, gm.equal_opportunity, threshold)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    frozen_ensemble = [
        FixedThresholdClassifier(
            FrozenEstimator(model), threshold=combined_fair_predictor.threshold
        )
        for model in ensemble_models
    ]

    end_time = time.perf_counter()
    training_time = end_time - start_time

    # Evaluate ensemble on test set
    logger.info("Evaluating ensemble on test set")

    # Get predictions from all ensemble members
    ensemble_preds = []
    test_predictions_data = {"data": test_data["data"], "target": test_data["target"]}
    for model in frozen_ensemble:
        preds = model.predict(test_predictions_data["data"])
        ensemble_preds.append(preds)

    # Calculate ensemble metrics
    final_preds = aggregate_scores(ensemble_preds)
    formatted_preds = consolidate_predictions(ensemble_preds)

    ensemble_test_metrics = calculate_metrics(
        test_groups=test_data["groups"],
        test_labels=test_data["target"],
        predictions=final_preds,
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
    for member, model in enumerate(frozen_ensemble):
        indiv_preds = model.predict(test_predictions_data["data"])
        indiv_metrics = calculate_metrics(
            test_groups=test_data["groups"],
            test_labels=test_data["target"],
            predictions=indiv_preds,
        )
        indiv_metrics["member"] = member
        indiv_metrics["split"] = "test"
        indiv_metrics["model_type"] = "individual"
        individual_metrics.append(indiv_metrics)

    return ensemble_test_metrics, individual_metrics, frozen_ensemble


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
    dataset: str = "adult",
):
    # Set up output directory
    output_dir = Path("./ensemble_results")
    output_dir.mkdir(exist_ok=True)

    # Load adult dataset with ethnicity as sensitive attribute
    logger.info(
        f"Loading {dataset.capitalize()} dataset with ethnicity as sensitive attribute"
    )

    replacement_groups = (
        {
            "Asian": "Other",
            "Native American": "Other",
        }
        if dataset == "compas"
        else {}
    )

    # Analyze original ethnicity distribution
    loader_func = datasets[dataset]
    train_data, val_data, test_data = loader_func(
        "race",
        train_proportion=0.6,
        test_proportion=0.2,  # Remaining 0.2 goes to validation
        seperate_groups=True,
        replace_groups=replacement_groups,
        seed=42,
    )

    ethnicity_dist = pd.Series(train_data["groups"]).value_counts()
    logger.info(f"Original ethnicity distribution: {ethnicity_dist}")

    # Create results dataframe
    results = []

    # Train ensemble for each grouping
    all_metrics = []
    single_metrics = []
    for iteration in tqdm(range(iterations), desc="Iterations", unit="iteration"):
        train_data, val_data, test_data = loader_func(
            "race",
            train_proportion=0.6,
            test_proportion=0.2,  # Remaining 0.2 goes to validation
            seperate_groups=True,
            replace_groups=replacement_groups,
            seed=42 + iteration,
        )
        # Train ensemble
        ensemble_metrics, individual_metrics, ensemble_models = train_ensemble(
            train_data,
            val_data,
            test_data,
            num_members=members,
            iteration_number=iteration,
            metric=metric,
            threshold=threshold,
        )

        # Add grouping info to metrics
        ensemble_metrics["num_groups"] = len(np.unique(test_data["groups"]))
        ensemble_metrics["iteration"] = iteration
        # Save to results
        results.append(ensemble_metrics)
        # Save individual metrics
        indiv_df = pd.DataFrame(individual_metrics)
        indiv_df["dataset"] = dataset
        indiv_df["iteration"] = iteration
        indiv_df.assign(fairness_metric=metric).to_csv(
            output_dir
            / f"individual_metrics-{dataset}-n{members}-i{iteration}-metric_{metric}.csv",
            index=False,
        )

        probas = [
            ensemble_model.predict_proba(test_data["data"])[:, 1]
            for ensemble_model in ensemble_models
        ]
        prediction_thresholds = np.linspace(0, 1, 100)
        for pred_threshold in prediction_thresholds:
            preds = aggregate_probas(probas, threshold=pred_threshold)
            metrics = calculate_metrics(
                test_groups=test_data["groups"],
                test_labels=test_data["target"],
                predictions=preds,
            )
            metrics["threshold"] = pred_threshold
            metrics["iteration"] = iteration
            all_metrics.append(metrics)
        single_proba = probas[:1]
        for pred_threshold in prediction_thresholds:
            preds = aggregate_probas(single_proba, threshold=pred_threshold)
            metrics = calculate_metrics(
                test_groups=test_data["groups"],
                test_labels=test_data["target"],
                predictions=preds,
            )
            metrics["threshold"] = pred_threshold
            metrics["iteration"] = iteration
            single_metrics.append(metrics)
    pd.DataFrame(single_metrics).to_csv(
        output_dir
        / f"single_threshold_metrics_{dataset}-n{members}-i{iterations}-metric_{metric}.csv",
        index=False,
    )
    pd.DataFrame(all_metrics).to_csv(
        output_dir
        / f"threshold_metrics_{dataset}-n{members}-i{iterations}-metric_{metric}.csv",
        index=False,
    )

    # Save overall results
    results_df = pd.DataFrame(results).assign(fairness_metric=metric)
    results_df.to_csv(
        output_dir
        / f"ensemble_results-{dataset}-n{members}-i{iterations}-metric_{metric}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ensemble on Adult dataset with ethnicity groups",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(datasets),
        default="adult",
        help="Dataset to use",
    )
    args = parser.parse_args()
    main(
        members=args.members,
        metric=args.metric,
        threshold=args.threshold,
        iterations=args.iterations,
        dataset=args.dataset,
    )
