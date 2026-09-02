"""Extract the middle MNIST training image and name it after its label."""

from pathlib import Path
import struct

from PIL import Image


DATASET_DIR = Path(__file__).parent / "mnist_dataset"
IMAGES_PATH = DATASET_DIR / "train-images.idx3-ubyte"
LABELS_PATH = DATASET_DIR / "train-labels.idx1-ubyte"


def main() -> None:
    with IMAGES_PATH.open("rb") as images_file:
        image_magic, image_count, rows, columns = struct.unpack(
            ">IIII", images_file.read(16)
        )

        if image_magic != 2051:
            raise ValueError(f"Unexpected image magic number: {image_magic}")

        # Zero-based index: for 60,000 records this selects record 30,001.
        middle_index = image_count // 2
        image_size = rows * columns
        images_file.seek(16 + middle_index * image_size)
        pixels = images_file.read(image_size)

    with LABELS_PATH.open("rb") as labels_file:
        label_magic, label_count = struct.unpack(">II", labels_file.read(8))

        if label_magic != 2049:
            raise ValueError(f"Unexpected label magic number: {label_magic}")
        if label_count != image_count:
            raise ValueError("Image and label counts do not match")

        labels_file.seek(8 + middle_index)
        label_data = labels_file.read(1)

    if len(pixels) != image_size or len(label_data) != 1:
        raise ValueError("The dataset ended before the middle record was read")

    label = label_data[0]
    output_path = Path(__file__).parent / f"{label}.png"
    Image.frombytes("L", (columns, rows), pixels).save(output_path)

    print(f"Dataset index: {middle_index}")
    print(f"Label: {label}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
