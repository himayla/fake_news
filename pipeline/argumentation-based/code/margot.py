import pandas as pd
import subprocess
import os
import json
import numpy as np
import shutil

TEMP_NAME = "temp" # Change when you run different jobs at the same time

def create_folders(name: str):
    """
    Function to create the necessary folders to use MARGOT.
        Args:
            name: Name of the dataset (options: fake_real, kaggle, liar)
        Returns: None
    """
    if os.path.exists(path_to_temp_dir):
        shutil.rmtree(path_to_temp_dir)
    os.makedirs(f"{path_to_temp_dir}/news/train")
    os.makedirs(f"{path_to_temp_dir}/news/test")

    os.makedirs(f"{path_to_temp_dir}/arguments/train")
    os.makedirs(f"{path_to_temp_dir}/arguments/test")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(f"{results_path}/{name}"):           
        os.makedirs(f"{results_path}/{name}")
    

def write_to_doc(clean_text: str, type: str) -> None:
    """
        Function to convert texts in the DataFrame to temporary text files.
        The function writes out one string to a file, where the filename is the row number (found by counter).
        Args:
            clean_text = string
            type = string with e.g.: train, test
        Returns:
            Creates a folder with news texts.
    """
    global counter
    with open(f"{path_to_temp_dir}/news/{type}/{counter}.txt", "a+") as text_file:
        text_file.write(clean_text)
    counter += 1
    
def extract_argumentation(name: str, type: str) -> pd.DataFrame:
    """
        Function calls MARGOT.
        It loads texts from temporary folder and calls MARGOT to extract the claim(s) and evidence(s) to another temporary folder.
        For every 100 texts, it writes out temporary results, this is counted by step.
        Args:
            name = string with e.g.: kaggle, fake_real, liar
            type = string with e.g.: train, test.
        Returns: 
            DataFrame with claim and evidence for each article. 
    """
    global step

    for idx, file in enumerate(sorted(os.listdir(f"{path_to_temp_dir}/news/{type}"), key=lambda x: int(x.split('.')[0]))):

        total = len(os.listdir(f"{path_to_temp_dir}/news/{type}"))
        print(f"STEP: {step}")
        print("------------------------------------------------------------------------")

        input = f"{TEMP_NAME}/news/{type}/{file}"
        output = f"{TEMP_NAME}/arguments/{type}/{idx}"
        subprocess.call([path_to_run, input, output])

        step += 1

        # Write out temporary results
        if step in np.arange(0, total, 100):
            print("WRITE OUT TEMPORARY RESULTS")
            print("------------------------------------------------------------------------")
            temp_result = parse_output(type)
            temp_result.to_csv(f"{results_path}/{name}/{type}.csv", index_label="ID")

    result = parse_output(type)
    return result

def parse_output(type: str) -> pd.DataFrame:
    """
        Function converts the temporary folder containing claim(s) and evidence(s) to a DataFrame.
        It goes through the temporary folder 'arguments' and adds the claim(s) and evidence(s) to a dictionary.
        It converts the dictionary to a DataFrame
        Args:
            type = string with e.g.: train, test.
        Returns:
            DataFrame containing the claim and evidences
            type = string with e.g.: train, test.
    """
    claim_evidence = {}
    for dir in sorted(os.listdir(f"{path_to_temp_dir}/arguments/{type}"), key=lambda x: int(x.split('.')[0])):
        with open(f"{path_to_temp_dir}/arguments/{type}/{dir}/OUTPUT.json") as json_file:
            doc = json.loads(json_file.read())

            options = {"claim": [], "evidence": []}

            for sent in doc["document"]:

                if sent.get("claim"):
                    s = sent["claim"] + "\""
                    options["claim"].append(s.replace(' .', '.'))

                elif sent.get("evidence"):
                    s = sent["evidence"]
                    options["evidence"].append(s.replace(' .', '.'))

        claim_evidence[dir] = options
    
    result = pd.DataFrame.from_dict(claim_evidence, orient='index')
 
    return result

def rejoin_data(original: str, 
                structure: str) -> dict[pd.DataFrame]:
    """
        Function combines annotated results to original.
        Args:

    """
    # Only continue with rows that either contain claim or evidence (not total empty)
    df_structure = structure.loc[(structure['claim'] != "[]") | (structure['evidence'] != "[]")]
    
    merged_df = pd.concat([original.reset_index(drop=True), df_structure.reset_index(drop=True)], axis=1)

    # Merge claim & evidence together as "structure"
    merged_df["claim"] = merged_df["claim"].apply(lambda x: ', '.join(x) if x else '<unk>')
    merged_df["evidence"] = merged_df["evidence"].apply(lambda x: ', '.join(x) if x else '<unk>')
    claims = "claims:" + merged_df["claim"].values
    evidences = "evidences:" + merged_df["evidence"].values

    merged_df["structure"] = claims + evidences
    merged_df.dropna(inplace=True)

    return merged_df

def run(original: pd.DataFrame, name: str, type: str) -> None:
    """
        Args:
            original = DataFrame before MARGOT
            name = string name of dataset. Choices are kaggle, fake_real, liar
    """
    original.apply(lambda x: write_to_doc(x["text"], type=type), axis=1)
    structure = extract_argumentation(name, type)
    result = rejoin_data(original, structure)

    result.to_csv(f"{results_path}/{name}/{type}.csv", columns=["ID", "text", "claim", "evidence", "structure", "label"])

if __name__ == "__main__":
    path_to_data = "pipeline/argumentation-based/data"
    path_to_temp_dir = f"pipeline/argumentation-based/tools/predictor/{TEMP_NAME}"

    path_to_margot = "pipeline/argumentation-based/tools/predictor"
    path_to_run = "pipeline/argumentation-based/tools/predictor/run_margot.sh"

    results_path = "pipeline/argumentation-based/argumentation structure/margot"

    for name in os.listdir(path_to_data): 
        if os.path.isdir(f"{path_to_data}/{name}") and name != ".DS_Store":

            create_folders(name)
            step, counter = 0, 0
    
            train = pd.read_csv(f"{path_to_data}/{name}/train.csv").dropna()
            test = pd.read_csv(f"{path_to_data}/{name}/test.csv").dropna()

            print(f"DATASET: {name} - LENGTH TRAIN: {len(train)}")
            print("------------------------------------------------------------------------\n")

            # Convert news values out to documents
            run(train, name, "train")
            run(test, name, "test")