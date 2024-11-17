import os
import pandas as pd
import shutil
import sys


def gender_to_class(gender):
    # M/F: 0/1
    return 0 if gender == 'M' else 1


def class_to_gender(class_label):
    # 0/1 -> M/F
    return 'F' if class_label % 2 else 'M'


def health_to_class(health):
    # H/PD: 0/2
    return 0 if health == 'H' else 2


def class_to_health(class_label):
    # 0/2 -> H/PD
    return 'H' if class_label < 2 else 'PD'


if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <metadata file>")
    sys.exit(1)

metadata_file = sys.argv[1]
assert os.path.isfile(metadata_file), f"Metadata file not found: {metadata_file}"
shutil.copy(metadata_file, f"{metadata_file}.bak")

metadata = pd.read_csv(metadata_file)
# Fill missing values
metadata["health"] = metadata["health"].fillna("H")
metadata["class"] = metadata["health"].apply(health_to_class) + metadata["gender"].apply(gender_to_class)
metadata.to_csv(metadata_file, index=False)
