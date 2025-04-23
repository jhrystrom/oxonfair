#!/usr/bin/env python3
import argparse
import os
import random
from pathlib import Path

import pandas as pd
import pytorch_lightning as L
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from dotenv import load_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

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
        "--train_csv",
        type=Path,
        default=Path("new_train.csv"),
        help="Relative path under data_dir to training metadata CSV",
    )
    parser.add_argument(
        "--val_csv",
        type=Path,
        default=Path("new_val.csv"),
        help="Relative path under data_dir to validation metadata CSV",
    )
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=Path("new_test.csv"),
        help="Relative path under data_dir to test metadata CSV",
    )
    parser.add_argument(
        "--img_dir",
        type=Path,
        default=Path("images/"),
        help="Relative path under data_dir to image directory",
    )
    parser.add_argument(
        "--path_col",
        type=str,
        default="Path",
        help="Name of the CSV column that holds each image filename",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        required=True,
        help="Index of the target label column in the CSV",
    )
    parser.add_argument(
        "--protected_class",
        type=int,
        required=True,
        help="Index of the protected label column in the CSV",
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
        type=int,
        default=1000,
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
        help="Scaling on the second-head loss",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for all loaders",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=3,
        help="Number of folds for stratified K-fold (default: 3)",
    )
    return parser.parse_args()


class CustomImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        path_col: str,
        target_idx: int,
        protected_idx: int,
        transform: transforms.Compose,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.path_col = path_col
        self.target_idx = target_idx
        self.protected_idx = protected_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self.img_dir / row[self.path_col]
        image = transforms.functional.pil_to_tensor(
            transforms.functional.load_image(str(img_path))
        )
        image = image.float() / 255.0
        if self.transform:
            image = self.transform(image)
        y_target = float(row.iloc[self.target_idx])
        y_prot = float(row.iloc[self.protected_idx])
        labels = torch.tensor([y_target, y_prot], dtype=torch.float32)
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
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    loss1 = bce(pred[:, 0], true[:, 0])
    loss2 = mse(pred[:, 1], true[:, 1])
    return loss1 + scaling * loss2


class LitTwoHead(L.LightningModule):
    def __init__(self, model: nn.Module, scaling: float, lr: float = 1e-4):
        super().__init__()
        self.model = model
        self.scaling = scaling
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        x, y = batch
        pred = self(x)
        loss = total_loss(pred, y, self.scaling)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        x, y = batch
        pred = self(x)
        loss = total_loss(pred, y, self.scaling)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def main():
    args = parse_args()

    # resolve full paths
    data_dir: Path = args.data_dir
    train_csv = data_dir / args.train_csv
    val_csv = data_dir / args.val_csv
    test_csv = data_dir / args.test_csv
    img_dir = data_dir / args.img_dir

    # load and merge train+val
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_merge = pd.concat([df_train, df_val], ignore_index=True)

    # load test separately
    df_test = pd.read_csv(test_csv)

    img_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    skf = StratifiedKFold(
        n_splits=args.num_folds, shuffle=True, random_state=RANDOM_SEED
    )
    stratify_vals = df_merge.iloc[:, args.target_class]

    # collect predictions
    preds_merge: dict[str, list[float]] = {}
    preds_test: dict[str, list[float]] = {}

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_merge, stratify_vals), start=1
    ):
        print(f"\n=== Fold {fold}/{args.num_folds} ===")

        train_ds = CustomImageDataset(
            df_merge.iloc[train_idx],
            img_dir,
            args.path_col,
            args.target_class,
            args.protected_class,
            transform=img_transform,
        )
        val_ds = CustomImageDataset(
            df_merge.iloc[val_idx],
            img_dir,
            args.path_col,
            args.target_class,
            args.protected_class,
            transform=img_transform,
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

        backbone = get_backbone(args.backbone)
        lit_model = LitTwoHead(backbone, args.scaling_factor)

        ckpt_dir = data_dir / "checkpoints" / f"fold{fold}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=ckpt_dir,
            filename="best-{epoch:02d}-{val_loss:.3f}",
            save_top_k=1,
        )
        logger = TensorBoardLogger(data_dir / "tb_logs", name=f"fold{fold}")

        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches or 1.0,
            callbacks=[checkpoint_cb],
            logger=logger,
            deterministic=True,
        )
        trainer.fit(lit_model, train_loader, val_loader)

        # load best checkpoint
        best_path = checkpoint_cb.best_model_path
        print("Loading checkpoint:", best_path)
        trained = LitTwoHead.load_from_checkpoint(
            best_path, model=get_backbone(args.backbone), scaling=args.scaling_factor
        )
        trained.eval()

        # predict on merged set
        full_merge_ds = CustomImageDataset(
            df_merge,
            img_dir,
            args.path_col,
            args.target_class,
            args.protected_class,
            transform=img_transform,
        )
        full_merge_loader = DataLoader(
            full_merge_ds, batch_size=args.batch_size, shuffle=False
        )

        preds_fold_merge: list[tuple[float, float]] = []
        with torch.no_grad():
            for x, _ in full_merge_loader:
                out = trained(x)
                p1 = torch.sigmoid(out[:, 0]).cpu().tolist()
                p2 = out[:, 1].cpu().tolist()
                preds_fold_merge.extend(zip(p1, p2))

        for h1, h2 in preds_fold_merge:
            preds_merge.setdefault(f"fold{fold}_h1", []).append(h1)
            preds_merge.setdefault(f"fold{fold}_h2", []).append(h2)

        # predict on test set
        test_ds = CustomImageDataset(
            df_test,
            img_dir,
            args.path_col,
            args.target_class,
            args.protected_class,
            transform=img_transform,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        preds_fold_test: list[tuple[float, float]] = []
        with torch.no_grad():
            for x, _ in test_loader:
                out = trained(x)
                p1 = torch.sigmoid(out[:, 0]).cpu().tolist()
                p2 = out[:, 1].cpu().tolist()
                preds_fold_test.extend(zip(p1, p2))

        for h1, h2 in preds_fold_test:
            preds_test.setdefault(f"fold{fold}_h1", []).append(h1)
            preds_test.setdefault(f"fold{fold}_h2", []).append(h2)

    # save merged predictions
    out_df_merge = pd.DataFrame({"index": df_merge.index})
    for col, vals in preds_merge.items():
        out_df_merge[col] = vals
    merged_out = data_dir / "raw_ensemble_predictions_trainval.csv"
    out_df_merge.to_csv(merged_out, index=False)
    print(f"Saved train+val predictions to {merged_out}")

    # save test predictions
    out_df_test = pd.DataFrame({"index": df_test.index})
    for col, vals in preds_test.items():
        out_df_test[col] = vals
    test_out = data_dir / "raw_ensemble_predictions_test.csv"
    out_df_test.to_csv(test_out, index=False)
    print(f"Saved test predictions to {test_out}")


if __name__ == "__main__":
    main()
