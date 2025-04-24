import os
import pickle
from pathlib import Path

import pandas as pd
import torchvision
import torchvision.transforms as transforms
from dotenv import load_dotenv
from tqdm import tqdm


def create_image_pickle(
    df: pd.DataFrame,
    img_dir: Path,
    path_col: str,
    batch_size: int = 100,
):
    """
    Process all images referenced in the DataFrame, apply transforms, and save to a pickle file.
    """
    # Initialize dictionary to store processed images
    image_dict = {}

    # Define transforms
    img_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Get unique image paths
    unique_paths = df[path_col].unique()

    print(f"Processing {len(unique_paths)} unique images...")

    # Process images in batches
    for i in tqdm(range(0, len(unique_paths), batch_size)):
        batch_paths = unique_paths[i : i + batch_size]

        for img_path_str in batch_paths:
            img_path = img_dir / img_path_str

            try:
                # Read and decode the image
                image = torchvision.io.decode_image(
                    torchvision.io.read_file(str(img_path))
                )

                # Normalize to [0, 1] range
                image = image.float() / 255.0

                # Apply transforms
                image = img_transform(image)

                # Store in dictionary
                image_dict[img_path_str] = image

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
    return image_dict


# Example usage:
if __name__ == "__main__":
    # Replace these with your actual values
    load_dotenv()
    directory = Path(os.getenv("PAPILA_PATH")) / "split"
    datasets = [directory / f"new_{split}.csv" for split in ["train", "val", "test"]]
    df = pd.concat([pd.read_csv(dataset) for dataset in datasets], ignore_index=True)
    img_dir = directory.parent / "FundusImages"
    path_col = "Path"  # Column containing image paths/filenames
    output_pickle_path = directory / Path("preprocessed_images.pkl")

    image_dict = create_image_pickle(
        df=df, img_dir=img_dir, path_col=path_col, output_pickle_path=output_pickle_path
    )
    output_pickle_path.write_bytes(pickle.dumps(image_dict))

    # Print some statistics
    print(f"Number of images in pickle: {len(image_dict)}")

    import time

    start = time.perf_counter_ns()
    loaded = pickle.loads(output_pickle_path.read_bytes())
    end = time.perf_counter_ns()
    print(f"Time to load pickle: {(end - start) / 1e6:.2f} ms")

    # Check the size of a few images
    for i, (key, img) in enumerate(list(image_dict.items())[:3]):
        print(f"Sample image {i+1}: {key}, shape: {img.shape}, dtype: {img.dtype}")
