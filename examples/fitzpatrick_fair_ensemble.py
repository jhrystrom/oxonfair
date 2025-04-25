import argparse
import copy
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
import torch.nn.functional as F
import torchvision.io
import torchvision.models as models
import torchvision.transforms as transforms

# Import wandb
import wandb
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import oxonfair for fairness-aware predictions
from oxonfair import group_metrics as gm

# fix random seeds
RANDOM_SEED = 4
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

load_dotenv(override=True)


def parse_args() -> argparse.Namespace:
    default_data_dir = Path(os.getenv("PAPILA_PATH", "."))  # fallback to cwd
    if default_data_dir == Path():
        print("Warning: PAPILA_PATH not set, using current directory as data_dir")
    parser = argparse.ArgumentParser(
        description="K-fold ensemble training of a two-headed model on a custom image dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=default_data_dir,
        help="Root data directory (default: $PAPILA_DIR)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("hatespeech-data/fitzpatrick17k.csv"),
        help="Path to CSV file with full data",
    )
    parser.add_argument(
        "--img_dir",
        type=Path,
        default=Path("Fitzpatrick17k/images"),
        help="Relative path under data_dir to image directory",
    )
    parser.add_argument(
        "--path_col",
        type=str,
        default="md5hash",
        help="Name of the CSV column that holds each image key",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="three_partition_label",
        help="Name of the target label column in the CSV",
    )
    parser.add_argument(
        "--protected_col",
        type=str,
        default="fitzpatrick_scale",
        help="Name of the protected label column in the CSV",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv3",
        choices=["mobilenetv3", "resnet18", "resnet50"],
        help="Which ImageNet backbone to use",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=1.0,
        help="How many training batches per epoch (useful for prototyping)",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=0.0,
        help="How many validation batches per epoch (0.0 = full)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=20,
        help="Number of epochs per fold",
    )
    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=0.5,
        help="Scaling on the attribute-head loss",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for all loaders",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=3,
        help="Number of folds for stratified K-fold (default: 3)",
    )
    # Add fairness-related arguments
    parser.add_argument(
        "--fairness_metric",
        type=str,
        default="demographic_parity",
        choices=[
            "demographic_parity",
            "equal_opportunity",
            "equalized_odds",
            "predictive_parity",
            "treatment_equality",
            "min_recall",
        ],
        help="Fairness metric to enforce with oxonfair",
    )
    parser.add_argument(
        "--fairness_threshold",
        type=float,
        default=0.05,
        help="Maximum allowed unfairness threshold (e.g., 0.05 means DP within 5%)",
    )
    parser.add_argument(
        "--performance_metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "balanced_accuracy", "precision", "recall"],
        help="Performance metric to optimize in the fairness-performance trade-off",
    )
    parser.add_argument(
        "--grid_width",
        type=int,
        default=75,
        help="Grid width for oxonfair's fair predictor search",
    )
    parser.add_argument(
        "--save_fair_models",
        action="store_true",
        help="Whether to save the fair versions of the models",
    )
    # Add wandb-related arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="fairness-image-classification",
        help="Name of the WandB project to log to",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity (username or team name)",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="WandB run name (defaults to auto-generated name)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run WandB in offline mode",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable WandB logging completely",
    )
    return parser.parse_args()


def custom_one_hot(x):
    # Create a new tensor with the mapped values
    # Map 1,2,3,4 -> 0; 5 -> 1; 6 -> 2
    mapped = torch.zeros_like(x)
    mapped = torch.where((x >= 1) & (x <= 4), torch.zeros_like(x), x)
    mapped = torch.where(x == 5, torch.ones_like(x), mapped)
    mapped = torch.where(x == 6, 2 * torch.ones_like(x), mapped)

    # One-hot encode the mapped values (with 3 classes)
    one_hot = F.one_hot(mapped.to(torch.int64), num_classes=3)

    return one_hot.float()  # Convert to float for gradient computation


class CustomImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dict: dict,  # Pre-loaded dictionary of images
        path_col: str,
        target_col: str,
        protected_col: str,
        transform_ops=None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dict = img_dict
        self.path_col = path_col
        self.target_col = target_col
        self.protected_col = protected_col
        self.transform_ops = transform_ops

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_key = row[self.path_col]

        # Get the pre-processed image directly from the dictionary
        image = self.img_dict[img_key]
        if self.transform_ops is not None:
            try:
                image = self.transform_ops(image)
            except OSError as e:
                print(f"Error applying transforms to image {img_key}: {e}")
                raise

        y_target = torch.tensor(float(row[self.target_col]))
        y_prot = custom_one_hot(torch.tensor(row[self.protected_col]))
        # labels of shape (batch_size, 1 + num_protected_classes)
        labels = torch.cat((y_target.unsqueeze(0), y_prot), dim=0)

        return image, labels


def get_backbone(name: str) -> nn.Module:
    if name == "mobilenetv3":
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        m.classifier[3] = nn.Linear(1024, 2)
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return m


def total_loss(pred: torch.Tensor, true: torch.Tensor, scaling: float) -> torch.Tensor:
    num_heads = pred.shape[1]
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    loss1 = bce(pred[:, 0], true[:, 0])
    loss2 = 0
    for i in range(1, num_heads):
        loss2 += mse(pred[:, i], true[:, i])
    return loss1 + scaling * loss2


class LitMultiHead(L.LightningModule):
    def __init__(self, model: nn.Module, scaling: float, lr: float = 1e-4):
        super().__init__()
        self.model = model
        self.scaling = scaling
        self.lr = lr
        self.train_acc = 0
        self.val_acc = 0
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        x, y = batch
        pred = self(x)
        loss = total_loss(pred, y, self.scaling)

        # Calculate accuracy for logging
        y_pred = torch.sigmoid(pred[:, 0]) > 0.5
        y_true = y[:, 0].bool()
        acc = (y_pred == y_true).float().mean()

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.train_acc = acc

        return loss

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        x, y = batch
        pred = self(x)
        loss = total_loss(pred, y, self.scaling)

        # Calculate accuracy for logging
        y_pred = torch.sigmoid(pred[:, 0]) > 0.5
        y_true = y[:, 0].bool()
        acc = (y_pred == y_true).float().mean()

        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.val_acc = acc

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def get_fairness_metric(name: str):
    """Map fairness metric name to oxonfair function"""
    metrics = {
        "demographic_parity": gm.demographic_parity,
        "equal_opportunity": gm.equal_opportunity,
        "equalized_odds": gm.equalized_odds,
        "predictive_parity": gm.predictive_parity,
        "treatment_equality": gm.treatment_equality,
        "min_recall": gm.recall.min,
    }
    return metrics.get(name)


def get_performance_metric(name: str):
    """Map performance metric name to oxonfair function"""
    metrics = {
        "accuracy": gm.accuracy,
        "balanced_accuracy": gm.balanced_accuracy,
        "precision": gm.precision,
        "recall": gm.recall,
    }
    return metrics.get(name)


def collect_predictions(model, loader, device):
    """Collect model predictions on the given loader"""
    model.eval()
    all_outputs = []
    all_targets = []
    all_groups = []

    with torch.no_grad():
        for x, y in loader:
            outputs = model(x.to(device)).cpu().numpy()
            all_outputs.append(outputs)
            all_targets.append(y[:, 0].numpy())  # First column is the target
            all_groups.append(
                y[:, 1].numpy()
            )  # Second column is the protected attribute

    return (
        np.concatenate(all_outputs),
        np.concatenate(all_targets),
        np.concatenate(all_groups),
    )


def read_image(path: Path) -> torch:
    return torchvision.io.decode_image(path)


def main() -> None:
    args = parse_args()
    TARGET_COL = args.target_col
    PROTECTED_COL = args.protected_col
    PATH_COL = args.path_col

    # Initialize wandb
    if not args.no_wandb:
        wandb_mode = "offline" if args.offline else "online"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=wandb_mode,
            config={
                "backbone": args.backbone,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "scaling_factor": args.scaling_factor,
                "fairness_metric": args.fairness_metric,
                "fairness_threshold": args.fairness_threshold,
                "performance_metric": args.performance_metric,
                "num_folds": args.num_folds,
                "random_seed": RANDOM_SEED,
            },
        )

    # pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")

    # resolve data paths
    data_dir: Path = Path(args.data_dir)
    train_csv = Path(args.csv)
    assert train_csv.exists(), f"CSV file {train_csv} does not exist"
    img_dir = data_dir / args.img_dir

    # From Fitz17k repo
    train_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # load CSVs
    df_full = pd.read_csv(train_csv)
    assert PATH_COL in df_full.columns, f"Path column {args.path_col} not found in CSV"
    assert TARGET_COL in df_full.columns, f"Target column {TARGET_COL} not found in CSV"
    assert (
        args.protected_col in df_full.columns
    ), f"Protected column {args.protected_col} not found in CSV"
    image_paths = list(img_dir.glob("*.jpg"))
    image_ids = {path.stem for path in image_paths}
    df_full = df_full[
        (df_full[PROTECTED_COL] != -1) & df_full[PATH_COL].isin(image_ids)
    ]
    df_full[TARGET_COL] = df_full[TARGET_COL] == "malignant"

    print(f"Found {len(df_full)} images in the CSV")
    if not args.no_wandb:
        wandb.log({"dataset_size": len(df_full)})

    assert (
        df_full[PATH_COL].isin(image_ids).all()
    ), "Some images are missing in the dictionary"
    img_dict = {path.stem: read_image(path) for path in tqdm(image_paths)}

    train_df, test_df = train_test_split(
        df_full, stratify=df_full[args.target_col], test_size=0.2
    )

    # stratified K-fold
    skf = StratifiedKFold(
        n_splits=args.num_folds,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    # containers
    preds_test: dict[str, list[float]] = {}
    fair_preds_test: dict[str, list[float]] = {}

    # Track fold metrics for final averaging
    fold_metrics = {
        "original_performance": [],
        "fair_performance": [],
        "original_fairness": [],
        "fair_fairness": [],
        "best_val_loss": [],
        "coefficients": [],
    }

    # Get the oxonfair metrics based on argument names
    fairness_metric = get_fairness_metric(args.fairness_metric)
    performance_metric = get_performance_metric(args.performance_metric)

    if fairness_metric is None:
        raise ValueError(f"Unknown fairness metric: {args.fairness_metric}")
    if performance_metric is None:
        raise ValueError(f"Unknown performance metric: {args.performance_metric}")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df[args.target_col]),
        start=1,
    ):
        print(f"\n=== Fold {fold}/{args.num_folds} ===")

        if not args.no_wandb and fold > 1:
            # Finish previous run
            wandb.finish()
            # Start new run for this fold
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{args.wandb_name or 'model'}_fold{fold}",
                group=args.wandb_name,
                job_type=f"fold_{fold}",
                mode="offline" if args.offline else "online",
                config={
                    "fold": fold,
                    "backbone": args.backbone,
                    "batch_size": args.batch_size,
                    "max_epochs": args.max_epochs,
                    "scaling_factor": args.scaling_factor,
                    "fairness_metric": args.fairness_metric,
                    "fairness_threshold": args.fairness_threshold,
                    "performance_metric": args.performance_metric,
                    "num_folds": args.num_folds,
                    "random_seed": RANDOM_SEED,
                },
            )

        # create & clean checkpoint dir
        ckpt_dir = data_dir / "checkpoints" / f"fold{fold}"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Create fair model directory if needed
        fair_dir = data_dir / "fair_models"
        if args.save_fair_models:
            fair_dir.mkdir(parents=True, exist_ok=True)

        # data loaders
        train_ds = CustomImageDataset(
            train_df.iloc[train_idx],
            img_dict,
            args.path_col,
            args.target_col,
            args.protected_col,
            transform_ops=train_transforms,
        )
        val_ds = CustomImageDataset(
            train_df.iloc[val_idx],
            img_dict,
            args.path_col,
            args.target_col,
            args.protected_col,
            transform_ops=test_transforms,
        )
        test_ds = CustomImageDataset(
            test_df,
            img_dict,
            args.path_col,
            args.target_col,
            args.protected_col,
            transform_ops=test_transforms,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # model + LightningModule
        backbone = get_backbone(args.backbone)
        lit_model = LitMultiHead(backbone, args.scaling_factor)

        # checkpoint & logger
        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=ckpt_dir,
            filename="best-{epoch:02d}-{val_loss:.3f}",
            save_top_k=1,
        )

        # Initialize WandB logger if not disabled
        logger = None
        if not args.no_wandb:
            logger = WandbLogger(
                project=args.wandb_project,
                log_model="all" if not args.offline else None,
                save_dir=str(data_dir / "wandb"),
            )

        # single-GPU trainer
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=args.max_epochs,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches or 1.0,
            callbacks=[checkpoint_cb],
            logger=logger,
            deterministic=True,
            log_every_n_steps=5,
        )
        trainer.fit(lit_model, train_loader, val_loader)
        print(f"[fold {fold}] best model: {checkpoint_cb.best_model_score:.3f}")

        # Log best validation loss
        fold_metrics["best_val_loss"].append(checkpoint_cb.best_model_score.item())
        if not args.no_wandb:
            wandb.log({"best_val_loss": checkpoint_cb.best_model_score.item()})

        trainer.strategy.barrier()

        # load the one fresh checkpoint
        best_path = checkpoint_cb.best_model_path
        print(f"[fold {fold}] loading checkpoint: {best_path}")
        trained = LitMultiHead.load_from_checkpoint(
            best_path,
            model=get_backbone(args.backbone),
            scaling=args.scaling_factor,
            map_location=device,
        )
        trained.eval().to(device)

    # After all folds, compute average metrics
    if not args.no_wandb:
        # Start a new summary run if we had multiple folds
        if args.num_folds > 1:
            wandb.finish()
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{args.wandb_name or 'model'}_summary",
                group=args.wandb_name or "model",
                job_type="summary",
                mode="offline" if args.offline else "online",
                config={
                    "backbone": args.backbone,
                    "batch_size": args.batch_size,
                    "max_epochs": args.max_epochs,
                    "scaling_factor": args.scaling_factor,
                    "performance_metric": args.performance_metric,
                    "num_folds": args.num_folds,
                    "random_seed": RANDOM_SEED,
                },
            )

        # Log average metrics across folds
        wandb.log(
            {
                "avg_val_loss": np.mean(fold_metrics["best_val_loss"]),
                "avg_original_performance": np.mean(
                    fold_metrics["original_performance"]
                ),
                "avg_fair_performance": np.mean(fold_metrics["fair_performance"]),
                "avg_original_fairness": np.mean(fold_metrics["original_fairness"]),
                "avg_fair_fairness": np.mean(fold_metrics["fair_fairness"]),
                "performance_drop": np.mean(fold_metrics["original_performance"])
                - np.mean(fold_metrics["fair_performance"]),
                "fairness_improvement": np.mean(fold_metrics["fair_fairness"])
                - np.mean(fold_metrics["original_fairness"]),
            }
        )

    out_df_test = pd.DataFrame({"index": test_df.index, **preds_test})
    test_out = data_dir / "raw_ensemble_predictions_test.csv"
    out_df_test.to_csv(test_out, index=False)
    print(f"Saved test predictions to {test_out}")

    fair_df_test = pd.DataFrame({"index": test_df.index, **fair_preds_test})
    fair_test_out = data_dir / "fair_ensemble_predictions_test.csv"
    fair_df_test.to_csv(fair_test_out, index=False)
    print(f"Saved fair test predictions to {fair_test_out}")

    # Create final visualizations and upload test predictions to wandb
    if not args.no_wandb:
        # Log test predictions as artifact
        if not args.offline:
            test_preds_artifact = wandb.Artifact(
                "test_predictions",
                type="predictions",
                description="Test set predictions from all folds",
            )
            test_preds_artifact.add_file(str(test_out))
            test_preds_artifact.add_file(str(fair_test_out))
            wandb.log_artifact(test_preds_artifact)

        # Create some final visualizations
        if args.num_folds > 1:
            # Plot fairness vs performance tradeoff across folds
            wandb.log(
                {
                    "fairness_performance_tradeoff": wandb.plot.scatter(
                        wandb.Table(
                            columns=["fold", "fairness", "performance", "type"],
                            data=[
                                [f + 1, fair, perf, "original"]
                                for f, (fair, perf) in enumerate(
                                    zip(
                                        fold_metrics["original_fairness"],
                                        fold_metrics["original_performance"],
                                    )
                                )
                            ]
                            + [
                                [f + 1, fair, perf, "fair"]
                                for f, (fair, perf) in enumerate(
                                    zip(
                                        fold_metrics["fair_fairness"],
                                        fold_metrics["fair_performance"],
                                    )
                                )
                            ],
                        ),
                        x="fairness",
                        y="performance",
                        title="Fairness vs Performance Tradeoff",
                    )
                }
            )

        # Finish the wandb run
        wandb.finish()


if __name__ == "__main__":
    main()
