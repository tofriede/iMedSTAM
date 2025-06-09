
import os
import glob
import math
import random
from argparse import ArgumentParser

random.seed(42)

TARGET_DISTRIBUTION = {
    'CT': 0.46,
    'MRI': 0.45,
    'US3D': 0.05,
    'PET': 0.035,
    'Microscopy': 0.005,
}

def main(full_dataset_dir: str):
    filepaths = glob.glob(full_dataset_dir + '/**/*.npz', recursive=True)
    filenames = [os.path.basename(filepath) for filepath in filepaths]

    full_dataset_size = len(filenames)
    coreset_size = math.floor(0.1 * full_dataset_size)



    distribution = {}
    for index, (modality, proportion) in enumerate(TARGET_DISTRIBUTION.items()):
        if index == len(TARGET_DISTRIBUTION) - 1:
            distribution[modality] = coreset_size - sum(distribution.values())
        else:
            distribution[modality] = math.floor(proportion * coreset_size)


    coreset_files = []

    print(f"Coreset size: {coreset_size}")
    print(distribution)

    for modality, modality_count in distribution.items():
        modality_dir = os.path.join(full_dataset_dir, modality)
        sub_datasets_file_counts = {}
        sub_datasets_files = {}

        # collect file counts and file paths for each sub-dataset
        for subdir in os.listdir(modality_dir):
            subdir_path = os.path.join(modality_dir, subdir)
            if os.path.isdir(subdir_path):
                files = [f for f in os.listdir(subdir_path) if f.endswith('.npz')]
                sub_datasets_file_counts[subdir] = len(files)
                sub_datasets_files[subdir] = [os.path.join(modality, subdir, f) for f in files]

        modality_size = sum(sub_datasets_file_counts.values())
        sorted_sub_datasets = sorted(sub_datasets_file_counts.items(), key=lambda x: x[1])

        # randomly sample files from each sub-dataset
        sampled_files = []
        for index, (subdir, file_count) in enumerate(sorted_sub_datasets):
            if file_count > 1:
                # ensure at least one file is sampled from each sub-dataset
                sampled_files.append(random.choice(sub_datasets_files[subdir]))

                # ensure the last sub-dataset has the remaining files
                if index == len(sorted_sub_datasets) - 1:
                    max_count = modality_count - len(sampled_files)
                    remaining_files = random.sample(sub_datasets_files[subdir], min(max_count, file_count - 1))
                else:
                    max_count = max(math.floor((file_count / modality_size) * modality_count), 1)
                    remaining_files = random.sample(sub_datasets_files[subdir], min(max_count - 1, file_count - 1))
                sampled_files.extend(remaining_files)
            else:
                sampled_files.extend(sub_datasets_files[subdir])  # include all files if only one exists

        coreset_files.extend(sampled_files)


    with open("training/coreset_files.txt", "w") as file:
        for i, filepath in enumerate(coreset_files):
            if i == len(coreset_files) - 1:
                file.write(filepath)
            else:
                file.write(filepath + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="path to the full dataset directory")
    args = parser.parse_args()
    main(args.data_dir)