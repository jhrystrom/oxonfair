import argparse
import copy
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, TypedDict

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F
from datasets import Dataset
from line_profiler import profile
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TextClassificationPipeline,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import (
    ModelOutput,  # or just use dict if not subclassing
)

import oxonfair
from oxonfair import group_metrics as gm


@dataclass
class MetricChoice:
    metric: gm.GroupMetric
    threshold: float


metric_choices = {
    "equal_opportunity": MetricChoice(metric=gm.equal_opportunity, threshold=0.02),
    "min_recall": MetricChoice(metric=gm.recall.min, threshold=0.75),
}


class FairnessMetrics(TypedDict):
    equal_opportunity: float
    min_recall: float
    accuracy: float
    precision: float
    recall: float


def calculate_metrics(
    test_groups: pl.Series, test_labels: pl.Series, predictions: list[str] | np.ndarray
) -> FairnessMetrics:
    groups = test_groups.to_numpy()
    preds0 = np.array(predictions)[groups == 0]
    preds1 = np.array(predictions)[groups == 1]
    labels0 = test_labels.to_numpy()[groups == 0]
    labels1 = test_labels.to_numpy()[groups == 1]

    recall1 = recall_score(y_true=labels1, y_pred=preds1)
    recall0 = recall_score(y_true=labels0, y_pred=preds0)

    min_recall = min(recall0, recall1)
    equal_opportunity = abs(recall1 - recall0)
    return {
        "min_recall": min_recall,
        "equal_opportunity": equal_opportunity,
        "accuracy": accuracy_score(y_true=test_labels.to_numpy(), y_pred=predictions),
        "precision": precision_score(y_true=test_labels, y_pred=predictions),
        "recall": recall_score(y_true=test_labels, y_pred=predictions),
    }


clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
CACHE_DIR = Path().cwd().parent / ".cache"
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(
        predictions=predictions, references=labels.astype(int).reshape(-1)
    )


def majority_vote(lists: list[list[bool]]) -> list[bool]:
    return [sum(sublist) > len(sublist) / 2 for sublist in lists]


# 8. Configure training arguments
training_args = TrainingArguments(
    output_dir="multilabel_model",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=256,
    fp16=True,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    label_names=["labels"],
)

# 3. Load tokenizer
model_path = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)


def convert_score(score: float, threshold: float = 0.5) -> bool:
    return score > threshold


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


def calculate_error_rate_avg(
    ensemble_preds: list[list[bool]], y_true: list[bool | int]
) -> float:
    num_members = len(ensemble_preds[0])
    total_error = 0
    for i in range(num_members):
        member_preds = [pred[i] for pred in ensemble_preds]
        member_error = 1 - accuracy_score(y_true=y_true, y_pred=member_preds)
        total_error += member_error
    return total_error / num_members


def calculate_der(preds: list[list[bool]], y_true: list[bool]) -> float:
    disagreement = calculate_disagreement_rates(preds).mean()
    average_error = calculate_error_rate_avg(ensemble_preds=preds, y_true=y_true)
    return disagreement / average_error


def calculate_der_groups(
    preds: list[list[bool]],
    y_true: list[bool],
    groups: list[bool],
    is_recall: bool = False,
) -> tuple[float, float]:
    group_mask = np.array(groups)
    ders = [0, 0]
    for group in range(2):
        mask_variable = group_mask == group
        if is_recall:
            logger.debug("Only calculating positive (recall)")
            mask_variable = mask_variable & np.array(y_true)
        y_true_group = np.array(y_true)[mask_variable == 1]
        group_indices = np.where(mask_variable)[0]
        group_preds = [pred for i, pred in enumerate(preds) if i in group_indices]
        assert (
            len(group_preds) == y_true_group.shape[0]
        ), f"{len(group_preds)=} != {y_true_group.shape[0]=}"
        group_der = calculate_der(preds=group_preds, y_true=y_true_group)
        ders[group] += group_der
    return ders[0], ders[1]


def aggregate_scores(scores: list[list[dict]], threshold: float = 0.5) -> list[bool]:
    final_preds = reformat_scores(scores, threshold)
    return majority_vote(final_preds)


def reformat_scores(scores: list[list[dict]], threshold: float = 0.5):
    num_preds = len(scores[0])
    # Convert to list[(pred0, pred0, pred0), (pred1, ...]
    final_preds = []
    for pred_index in range(num_preds):
        pred_list = []
        for score_list in scores:
            score_dict = score_list[pred_index]
            pred_list.append(convert_score(score_dict["score"], threshold=threshold))
        final_preds.append(pred_list)
    return final_preds


def max_index_by_key(lst: list[dict], key: str = "score"):
    if not lst:
        return None
    return max(range(len(lst)), key=lambda i: lst[i][key])


def ensemble_predict(
    texts: list[str], ensemble: list[Trainer], batch_size: int | None = None
) -> list[int]:
    preds = ensemble_predict_raw(texts, ensemble, batch_size)
    return aggregate_scores(preds)


def ensemble_predict_raw(
    texts: list[str], ensemble: list[Trainer], batch_size: int | None = None
):
    device = ensemble[0].model.device  # Get device from first model
    pipes = [
        TextClassificationPipeline(
            tokenizer=tokenizer,
            model=trainer.model.to(device),
            device=device,
            truncation=True,
        )
        for trainer in ensemble
    ]
    if batch_size is None:
        preds = [pipe(texts) for pipe in pipes]
    else:
        preds = []
        for pipe in tqdm(pipes):
            inner_preds = []
            for output in pipe(texts, batch_size=batch_size):
                inner_preds.append(output)
            preds.append(inner_preds)
    return preds


def get_full_data():
    english_hatespeech = Path().cwd().parent / "hatespeech-data" / "split" / "English"
    all_data = list(english_hatespeech.glob("*.tsv"))
    return (
        pl.DataFrame(
            pd.concat([pd.read_csv(f, sep="\t") for f in all_data]).drop(
                columns=["city", "state", "country", "date"]
            )
        )
        .with_columns(
            pl.col("gender").replace("x", None).cast(pl.Int8),
            pl.col("age").replace("x", None).cast(pl.Int8),
            pl.col("ethnicity").replace("x", None).cast(pl.Int8),
        )
        .drop_nulls()
        .rename({"label": "target"})
    )


def create_dataset(
    features: pl.DataFrame, labels: pl.Series, feature_names: list[str] | None = None
) -> Dataset:
    if feature_names is None:
        feature_names = features.columns
    feature_dict = {feature: features[feature].to_list() for feature in feature_names}
    return Dataset.from_dict(
        {
            **feature_dict,
            "target": labels.to_list(),
        }
    )


@lru_cache
def tokenize(text: str) -> dict[str, Any]:
    return tokenizer(text, truncation=True)


def preprocess_simple(example: dict[str, Any]) -> dict[str, Any]:
    tokenized = tokenize(example["text"])
    labels = [float(example[key]) for key in ["target", "gender"]]
    tokenized["labels"] = labels
    return tokenized


def compute_loss_func(
    outputs: ModelOutput | dict,
    labels: torch.Tensor,
    num_items_in_batch: int,  # noqa: ARG001
) -> torch.Tensor:
    """
    Custom loss function for HuggingFace Trainer:
    - Binary log loss for the first element
    - Squared loss (MSE) for the remaining elements

    Args:
        outputs: ModelOutput or dict containing 'logits' of shape (batch_size, num_outputs)
        labels: Tensor of shape (batch_size, num_outputs), ground-truth labels
        num_items_in_batch: Total number of items in the accumulated batch (unused here)
        num_classification_labels: Number of non-group based classification labels (default: 2)

    Returns:
        Scalar tensor representing the combined loss
    """
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

    log_loss = F.binary_cross_entropy_with_logits(logits[:, :1], labels[:, :1])

    # Regression loss (MSE) for remaining outputs
    if logits.shape[1] > 1:
        mse_loss = F.mse_loss(logits[:, 1:], labels[:, 1:])
        loss = log_loss + mse_loss
    else:
        loss = log_loss

    return loss


combined = get_full_data()  # .sample(fraction=0.2)

# 5. Prepare data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# 6. Metrics function
# 7. Initialize model


@profile
def train(metric: str, num_iterations: int = 1, members: int = 3):
    K = num_iterations
    NUM_MEMBERS = members
    gss = GroupShuffleSplit(n_splits=K, train_size=0.8, random_state=110)
    all_features = combined.drop("target", "tid", "uid", "age", "ethnicity")
    all_labels = combined["target"]
    all_users = combined["uid"]

    metric_choice = metric_choices[metric]

    test_metrics = []
    group_metrics = []
    all_metrics: list[pd.DataFrame] = []
    for iteration, (train_index, test_index) in tqdm(
        enumerate(gss.split(all_features, all_labels, groups=all_users))
    ):
        train_features = all_features[train_index]
        train_labels = all_labels[train_index]
        train_groups = all_users[train_index]
        test_features = all_features[test_index]
        test_labels = all_labels[test_index]

        test_dataset = create_dataset(
            test_features,
            test_labels,
        ).map(preprocess_simple)

        # nested cross-validation
        # Run oxonfair on an outer
        # Min recall as a key metric for each test partition
        # Key question: how big do we need to make the delta min recall to matter on the text
        fair_ensemble = []
        metrics = []
        oxons = []
        inner_gss = GroupShuffleSplit(
            n_splits=NUM_MEMBERS, train_size=0.8, random_state=110
        )
        for i, (inner_train_index, validation_index) in enumerate(
            inner_gss.split(train_features, train_labels, groups=train_groups)
        ):
            inner_train_features = train_features[inner_train_index]
            inner_train_labels = train_labels[inner_train_index]
            inner_train_groups = train_groups[inner_train_index]
            inner_validation_features = train_features[validation_index]
            inner_validation_labels = train_labels[validation_index]
            inner_validation_groups = train_groups[validation_index]
            assert inner_validation_groups.shape[0] == validation_index.shape[0]
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2,
                problem_type="multi_label_classification",
            )

            train_dataset = create_dataset(
                inner_train_features,
                inner_train_labels,
            ).map(preprocess_simple)

            validation_dataset = create_dataset(
                inner_validation_features,
                inner_validation_labels,
            ).map(preprocess_simple)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_loss_func=compute_loss_func,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            # Run oxonfair here? to merge heads etc.
            val_output = trainer.predict(validation_dataset)
            fpred = oxonfair.DeepFairPredictor(
                inner_validation_labels.to_numpy(),
                val_output.predictions,
                groups=np.array(validation_dataset["gender"]),
            )
            fpred.fit(
                gm.accuracy,
                metric_choice.metric,
                metric_choice.threshold,
                grid_width=75,
            )
            fpred.plot_frontier()
            plt.savefig(f"../plots/{metric}_val_{iteration}_{i}.png")
            fair_network = copy.deepcopy(trainer)
            fair_network.model.classifier = fpred.merge_heads_pytorch(
                fair_network.model.classifier
            )
            performance = fpred.evaluate().assign(
                classifier=i, metric_type="performance"
            )
            fairness = fpred.evaluate_fairness(
                metrics=gm.default_fairness_measures | {"min_recall": gm.recall.min}
            ).assign(classifier=i, metric_type="fairness")
            group_performance = (
                fpred.evaluate_groups()
                .reset_index()
                .assign(classifier=i, metric_type="performance")
            )
            group_metrics.append(group_performance)
            metrics.append(pd.concat([performance, fairness]))
            fair_ensemble.append(fair_network)

            # Evaluating on test set!
            test_output = trainer.predict(test_dataset=test_dataset)
            test_network = oxonfair.DeepDataDict(
                test_labels.to_numpy(),
                test_output.predictions,
                np.array(test_features["gender"]),
            )

            plt.clf()
            fpred.plot_frontier(data=test_network)
            plt.savefig(f"../plots/{metric}_test_{iteration}_{i}.png")
            plt.clf()

        pd.concat(group_metrics).to_csv(
            f"../hatespeech-data/group_metrics_iteration{iteration}.csv", index=False
        )

        all_metrics.append(pd.concat(metrics).assign(iteration=iteration))
        logger.info("Done training ensemble! Evaluating on test set")

        logger.debug("Evaluating ensemble...")
        raw_preds = ensemble_predict_raw(
            texts=test_dataset["text"], ensemble=fair_ensemble, batch_size=256
        )
        formatted_preds = reformat_scores(raw_preds)
        ensemble_der_total = calculate_der(
            preds=formatted_preds, y_true=test_labels.to_list()
        )

        der0, der1 = calculate_der_groups(
            preds=formatted_preds,
            y_true=test_labels.to_list(),
            groups=test_features["gender"].to_list(),
            is_recall=True,
        )

        der0complete, der1complete = calculate_der_groups(
            preds=formatted_preds,
            y_true=test_labels.to_list(),
            groups=test_features["gender"].to_list(),
            is_recall=False,
        )

        logger.info(f"{ensemble_der_total=}")
        logger.info("DER for true cases:")
        logger.info(f"{der0=} and {der1=}")
        logger.info("DER for all cases:")
        logger.info(f"{der0complete=} and {der1complete=}")

        ensemble_preds = aggregate_scores(raw_preds)

        ensemble_metrics = calculate_metrics(
            test_groups=test_features["gender"],
            test_labels=test_labels,
            predictions=ensemble_preds,
        )
        logger.debug("Evaluating first member...")
        single_preds = ensemble_predict(
            texts=test_dataset["text"], ensemble=fair_ensemble[:1], batch_size=256
        )
        single_metrics = calculate_metrics(
            test_groups=test_features["gender"],
            test_labels=test_labels,
            predictions=single_preds,
        )
        single_df = pd.DataFrame([single_metrics]).assign(model_type="single")
        test_metric_df = pd.concat(
            [single_df, pd.DataFrame([ensemble_metrics]).assign(model_type="ensemble")]
        ).assign(iteration=iteration)
        test_metrics.append(test_metric_df)
    return pd.concat(test_metrics), pd.concat(all_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric", type=str, choices=list(metric_choices), default="equal_opportunity"
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--members", type=int, default=1)
    args = parser.parse_args()
    test_metrics, all_metrics = train(
        metric=args.metric, num_iterations=args.iterations, members=args.members
    )
