import subprocess
import os
import csv
import random
import re

DATASET = "paultimothymooney/breast-histopathology-images"

IMAGES_PER_CLASS = 500

CLASS0_DIR = os.path.join("dataset", "class0")
CLASS1_DIR = os.path.join("dataset", "class1")

os.makedirs(CLASS0_DIR, exist_ok=True)
os.makedirs(CLASS1_DIR, exist_ok=True)


def get_page(page_token=None):

    command = [
        "python", "-m", "kaggle",
        "datasets", "files",
        DATASET,
        "--page-size", "200",
        "--csv"
    ]

    if page_token:
        command.extend(["--page-token", page_token])

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Error while getting Kaggle files:")
        print(result.stderr)
        return "", ""

    return result.stdout, result.stderr


def find_images():

    class0 = []
    class1 = []

    page_token = None
    page_number = 1

    print("Searching Kaggle for images...")
    print()

    while len(class0) < IMAGES_PER_CLASS or len(class1) < IMAGES_PER_CLASS:

        print(f"Reading page {page_number}...")

        stdout, stderr = get_page(page_token)

        if not stdout:
            print("No more results found.")
            break

        lines = stdout.splitlines()

        # Find CSV header
        csv_start = None

        for i, line in enumerate(lines):

            if line.lower().startswith("name,"):
                csv_start = i
                break

        if csv_start is not None:

            csv_lines = lines[csv_start:]

            reader = csv.DictReader(csv_lines)

            for row in reader:

                name = row.get("name", "")

                if not name.lower().endswith(".png"):
                    continue

                if "class0.png" in name.lower():

                    if name not in class0:
                        class0.append(name)

                elif "class1.png" in name.lower():

                    if name not in class1:
                        class1.append(name)

        print(
            f"Found class0: {len(class0)} | "
            f"class1: {len(class1)}"
        )

        # Kaggle prints the next token as:
        # Next Page Token = XXXXX

        combined_output = stdout + "\n" + stderr

        match = re.search(
            r"Next Page Token\s*=\s*(\S+)",
            combined_output,
            re.IGNORECASE
        )

        if match:

            page_token = match.group(1)
            page_number += 1

        else:

            print("No more pages found.")
            break

    return class0, class1


def download_images(files, destination, class_name):

    print()
    print(
        f"Downloading {len(files)} "
        f"{class_name} images..."
    )
    print()

    for i, file_name in enumerate(files, start=1):

        print(
            f"{class_name}: "
            f"{i}/{len(files)}"
        )

        subprocess.run([
            "python", "-m", "kaggle",
            "datasets", "download",
            DATASET,
            "-f", file_name,
            "-p", destination,
            "--unzip",
            "-q"
        ])


def main():

    class0, class1 = find_images()

    print()
    print("--------------------------------")
    print("Search completed")
    print("Class 0 found:", len(class0))
    print("Class 1 found:", len(class1))
    print("--------------------------------")

    if len(class0) < IMAGES_PER_CLASS:

        print(
            "Not enough class 0 images were found."
        )
        return

    if len(class1) < IMAGES_PER_CLASS:

        print(
            "Not enough class 1 images were found."
        )
        return

    random.seed(42)

    random.shuffle(class0)
    random.shuffle(class1)

    class0 = class0[:IMAGES_PER_CLASS]
    class1 = class1[:IMAGES_PER_CLASS]

    download_images(
        class0,
        CLASS0_DIR,
        "class0"
    )

    download_images(
        class1,
        CLASS1_DIR,
        "class1"
    )

    print()
    print("================================")
    print("Dataset download completed!")
    print("================================")


if __name__ == "__main__":
    main()