import argparse
import csv
import itertools
import os
import shutil

import pandas as pd


def augment_nemo_data(source_dir, target_dir, num_mixed=100):
    """
    Augments Training data to include more multi-label utterances by through utterance combining. 

    Args:
        source_dir: directory that contains nemo-format files
        target_dir: directory to store the newly transformed files
        num_mixed: the number of additional combined examples per class combination
    """

    train_df = pd.read_csv(f'{source_dir}/train.tsv', sep="\t")

    # Filler Slots
    slots_df = pd.read_csv(f'{source_dir}/train_slots.tsv', sep="\t", header=None)
    slots_df.columns = ["slots"]

    # Get Slots Dictionary
    slots_dict = {}
    slot_file = f'{source_dir}/dict.slots.csv'
    o_slot_index = 0

    with open(slot_file, "r") as f:
        slot_lines = f.readlines()

    dataset = list(slot_lines)

    # Find the index that contains the 'O' slot
    for i in range(len(dataset)):
        if dataset[i] == "O":
            o_slot_index = i
            break

    labels = train_df.columns[1:]
    actual_labels = train_df[labels].values.tolist()
    sentences = train_df['sentence'].values.tolist()
    results = []

    # Set of all existing lables
    all_labels = set(map(lambda labels: tuple(labels), actual_labels))

    label_indices = []

    for label in all_labels:
        label_indices.append([i for i, x in enumerate(actual_labels) if tuple(x) == label])

    series_list = []
    slots_list = []

    for i in range(len(label_indices)):
        for j in range(i + 1, len(label_indices)):
            # We now have two label indices:
            n = len(label_indices[i])
            m = len(label_indices[j])

            first_class_indices = label_indices[i]
            second_class_indices = label_indices[j]
            combined_list = list(itertools.product(first_class_indices, second_class_indices))
            combined_list = combined_list[: min(num_mixed, len(combined_list))]

            for index, index2 in combined_list:
                sentence1 = sentences[index]
                sentence2 = sentences[index2]

                labels1 = set(actual_labels[index][0].split(','))
                labels2 = set(actual_labels[index2][0].split(','))

                slots1 = slots_df["slots"][index]
                slots2 = slots_df["slots"][index2]

                combined_labels = ",".join(sorted(labels1.union(labels2)))
                combined_sentences = f"{sentence1}. {sentence2}"
                combined_lst = [combined_sentences] + [combined_labels]
                combined_slots = f"{slots1} {o_slot_index}  {slots2}"

                series_list.append(combined_lst)
                slots_list.append(combined_slots)

    new_df = pd.DataFrame(series_list, columns=train_df.columns)
    new_slots_df = pd.DataFrame(slots_list, columns=slots_df.columns)

    train_df = train_df.append(new_df)
    slots_df = slots_df.append(new_slots_df)
    train_df = train_df.reset_index(drop=True)
    slots_df = slots_df.reset_index(drop=True)
    train_df.to_csv(f'{target_dir}/train.tsv', sep="\t", index=False)
    slots_df.to_csv(f'{target_dir}/train_slots.tsv', sep="\t", index=False, header=False)


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description="Process and convert datasets into NeMo\'s format.")
    parser.add_argument(
        "--source_data_dir", required=True, type=str, help='path to the folder containing the dataset files'
    )
    parser.add_argument("--target_data_dir", required=True, type=str, help='path to save the processed dataset')
    parser.add_argument("--num_mixed", type=str, help='Number of training examples per class to mix')

    args = parser.parse_args()

    source_dir = args.source_data_dir
    target_dir = args.target_data_dir
    num_mixed = args.num_mixed

    augment_nemo_data(f'{source_dir}', f'{target_dir}', int(num_mixed))
    shutil.copyfile(f'{source_dir}/dict.intents.csv', f'{target_dir}/dict.intents.csv')
    shutil.copyfile(f'{source_dir}/dict.slots.csv', f'{target_dir}/dict.slots.csv')
    shutil.copyfile(f'{source_dir}/dev.tsv', f'{target_dir}/dev.tsv')
    shutil.copyfile(f'{source_dir}/dev_slots.tsv', f'{target_dir}/dev_slots.tsv')
