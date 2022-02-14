import argparse
import os
import shutil
from os.path import exists

import pandas as pd


def convert_atis_multi_label(path, source_dir, target_dir, mode):
    """
    converts atis data to multi-label data
    path - path to tsv file
    """
    data = pd.read_csv(path, sep='\t')
    # Get the original intent dictionary
    intents = pd.read_csv(f'{source_dir}/dict.intents.csv', sep='\t', header=None)
    # Create a new intent dictionary
    old_intents = []
    #  Get the original intents
    for index, intent in intents.iterrows():
        old_intents.append(intent[0])

    # Create new intent list mapping
    new_intents = [x for x in old_intents if "+" not in x]

    # New dictionary to convert to dataframe
    intent_labels = []

    for index, intent in data.iterrows():
        temp_dict = {}
        temp_dict['sentence'] = intent['sentence']
        old_label = old_intents[int(intent['label'])]

        values = [old_label]

        if '+' in old_label:
            values = old_label.split('+')

        for index, label in enumerate(new_intents):
            if label in values:
                if 'label' not in temp_dict:
                    temp_dict['label'] = f"{index}"
                else:
                    temp_dict['label'] = f"{temp_dict['label']},{index}"

        intent_labels.append(temp_dict)

    multi_intent_df = pd.DataFrame.from_dict(intent_labels)
    multi_intent_df.to_csv(f'{target_dir}/{mode}.tsv', sep='\t', index=False)


def convert_intent_dictionary(source_dir, target_dir):
    intents = pd.read_csv(f'{source_dir}/dict.intents.csv', sep='\t', header=None)

    new_atis_dict_list = []

    for index, intent in intents.iterrows():
        new_atis_dict_list.append(intent[0])

    new_intents = [x for x in new_atis_dict_list if "+" not in x]
    df = pd.DataFrame(new_intents)
    df.to_csv(f"{target_dir}/dict.intents.csv", index=False, header=False)


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description="Process and convert datasets into NeMo\'s format.")
    parser.add_argument(
        "--source_data_dir", required=True, type=str, help='path to the folder containing the dataset files'
    )
    parser.add_argument("--target_data_dir", required=True, type=str, help='path to save the processed dataset')

    args = parser.parse_args()

    source_dir = args.source_data_dir
    target_dir = args.target_data_dir

    convert_intent_dictionary(f'{source_dir}', f'{target_dir}')
    convert_atis_multi_label(
        f'{source_dir}/train.tsv', f'{source_dir}', f'{target_dir}', 'train'
    )
    convert_atis_multi_label(
        f'{source_dir}/test.tsv', f'{source_dir}', f'{target_dir}', 'dev'
    )
    shutil.copyfile(f'{source_dir}/dict.slots.csv', f'{target_dir}/dict.slots.csv')
    shutil.copyfile(f'{source_dir}/train_slots.tsv', f'{target_dir}/train_slots.tsv')
    shutil.copyfile(f'{source_dir}/test_slots.tsv', f'{target_dir}/dev_slots.tsv')
