import copy
from datetime import datetime
from functools import lru_cache
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
    per_device_train_batch_size=128,
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


def aggregate_scores(scores: list[list[dict]], threshold: float = 0.5) -> list[bool]:
    num_preds = len(scores[0])
    # Convert to list[(pred0, pred0, pred0), (pred1, ...]
    final_preds = []
    for pred_index in range(num_preds):
        pred_list = []
        for score_list in scores:
            score_dict = score_list[pred_index]
            pred_list.append(convert_score(score_dict["score"], threshold=threshold))
        final_preds.append(pred_list)
    return majority_vote(final_preds)


def max_index_by_key(lst: list[dict], key: str = "score"):
    if not lst:
        return None
    return max(range(len(lst)), key=lambda i: lst[i][key])


def ensemble_predict(
    texts: list[str], ensemble: list[Trainer], batch_size: int | None = None
) -> list[int]:
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
    return aggregate_scores(preds)


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


combined = get_full_data().sample(fraction=0.2)

# 5. Prepare data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# 6. Metrics function
# 7. Initialize model


@profile
def train():
    K = 1
    NUM_MEMBERS = 3
    gss = GroupShuffleSplit(n_splits=K, train_size=0.8, random_state=110)
    all_features = combined.drop("target", "tid", "uid", "age", "ethnicity")
    all_labels = combined["target"]
    all_users = combined["uid"]

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
            fpred.fit(gm.accuracy, gm.equal_opportunity, 0.02, grid_width=75)
            fpred.plot_frontier()
            plt.savefig(f"../plots/equal_opportunity_val_{iteration}_{i}.png")
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
            logger.debug({f"{group_performance.head()=}"})
            group_metrics.append(group_performance)
            metrics.append(pd.concat([performance, fairness]))
            fair_ensemble.append(fair_network)
            oxons.append(fpred)

        pd.concat(group_metrics).to_csv(
            f"../hatespeech-data/group_metrics_iteration{iteration}.csv", index=False
        )

        all_metrics.append(pd.concat(metrics).assign(iteration=iteration))
        logger.info("Done training ensemble! Evaluating on test set")
        test_dataset = create_dataset(
            test_features,
            test_labels,
        ).map(preprocess_simple)

        logger.debug("Evaluating ensemble...")
        ensemble_preds = ensemble_predict(
            texts=test_dataset["text"], ensemble=fair_ensemble, batch_size=256
        )
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
    test_metrics, all_metrics = train()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    test_metrics.to_csv(
        f"../hatespeech-data/test_metrics-{current_time}.csv", index=False
    )
    all_metrics.to_csv(
        f"../hatespeech-data/validation_metrics-{current_time}.csv", index=False
    )
