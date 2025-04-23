from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # read metadata
    path = Path("your_path/data/PAPILA/")

    # OD for right, OS for left
    od_meta = pd.read_csv(path / "ClinicalData/patient_data_od.csv")
    os_meta = pd.read_csv(path / "ClinicalData/patient_data_os.csv")

    # Create path columns for each eye dataset
    ids = os_meta["ID"].values
    os_path = ["RET" + x[1:] + "OS.jpg" for x in ids]
    os_meta["Path"] = os_path

    ids = od_meta["ID"].values
    od_path = ["RET" + x[1:] + "OD.jpg" for x in ids]
    od_meta["Path"] = od_path

    # Combine datasets and select relevant columns
    meta_all = pd.concat([od_meta, os_meta])
    subcolumns = ["ID", "Age", "Gender", "Diagnosis", "Path"]
    meta_all = meta_all[subcolumns]

    # Save the combined metadata
    meta_all.to_csv(path / "ClinicalData/patient_meta_concat.csv")

    # Convert gender codes to M/F
    # The patient (0 for male and 1 for female)
    # The diagnosis (0 stands for healthy, 1 for glaucoma, and 2 for suspicious)
    sex = meta_all["Gender"].values.astype("str")
    sex[sex == "0.0"] = "M"
    sex[sex == "1.0"] = "F"
    meta_all["Sex"] = sex

    # Create age categories
    # Multi-category age groups
    meta_all["Age_multi"] = meta_all["Age"].values.astype("int")
    meta_all["Age_multi"] = np.where(
        meta_all["Age_multi"].between(0, 19), 0, meta_all["Age_multi"]
    )
    meta_all["Age_multi"] = np.where(
        meta_all["Age_multi"].between(20, 39), 1, meta_all["Age_multi"]
    )
    meta_all["Age_multi"] = np.where(
        meta_all["Age_multi"].between(40, 59), 2, meta_all["Age_multi"]
    )
    meta_all["Age_multi"] = np.where(
        meta_all["Age_multi"].between(60, 79), 3, meta_all["Age_multi"]
    )
    meta_all["Age_multi"] = np.where(
        meta_all["Age_multi"] >= 80, 4, meta_all["Age_multi"]
    )

    # Binary age groups (younger vs older)
    meta_all["Age_binary"] = meta_all["Age"].values.astype("int")
    meta_all["Age_binary"] = np.where(
        meta_all["Age_binary"].between(0, 60), 0, meta_all["Age_binary"]
    )
    meta_all["Age_binary"] = np.where(
        meta_all["Age_binary"] >= 60, 1, meta_all["Age_binary"]
    )

    # Binary classification dataset (only healthy and glaucoma)
    meta_binary = meta_all[
        (meta_all["Diagnosis"].values == 1.0) | (meta_all["Diagnosis"].values == 0.0)
    ]
    print(f"Binary classification dataset size: {len(meta_binary)}")

    # Split data into train/val/test sets (70%/10%/20%)
    train_meta, val_meta, test_meta = split_712(
        meta_binary, np.unique(meta_binary["ID"])
    )

    # Save splits
    output_dir = path / "split"
    output_dir.mkdir(exist_ok=True, parents=True)

    train_meta.to_csv(output_dir / "new_train.csv")
    val_meta.to_csv(output_dir / "new_val.csv")
    test_meta.to_csv(output_dir / "new_test.csv")


def split_712(all_meta, patient_ids):
    """
    Split data into train/validation/test sets with a 70%/10%/20% ratio
    ensuring patient IDs aren't split across sets
    """
    sub_train, sub_val_test = train_test_split(
        patient_ids, test_size=0.3, random_state=5
    )
    sub_val, sub_test = train_test_split(sub_val_test, test_size=0.66, random_state=15)

    train_meta = all_meta[all_meta.ID.isin(sub_train)]
    val_meta = all_meta[all_meta.ID.isin(sub_val)]
    test_meta = all_meta[all_meta.ID.isin(sub_test)]

    return train_meta, val_meta, test_meta


if __name__ == "__main__":
    main()
