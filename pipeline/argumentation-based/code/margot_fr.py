import pandas as pd
import subprocess
import os
import json
import numpy as np
import shutil

DATASET = 'fake_real'
TEMP_NAME = "temp_fr_v"
BATCH_SIZE = 100

path_to_data = "pipeline/argumentation-based/data"
path_to_temp_dir = f"pipeline/argumentation-based/tools/predictor/{TEMP_NAME}"

path_to_margot = "pipeline/argumentation-based/tools/predictor"
path_to_run = "pipeline/argumentation-based/tools/predictor/run_margot.sh"

results_path = "pipeline/argumentation-based/argumentation structure/margot"

def create_folders(dataset: str, type: str):
    """
    Function to create the necessary folders to use MARGOT.
        Args:
            name: Name of the dataset (options: fake_real, kaggle, liar)
        Returns: None
    """
    if os.path.exists(path_to_temp_dir):
        shutil.rmtree(path_to_temp_dir)
    os.makedirs(f"{path_to_temp_dir}/{type}/news")
    os.makedirs(f"{path_to_temp_dir}/{type}/arguments")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(f"{results_path}/{dataset}"):           
        os.makedirs(f"{results_path}/{dataset}")
    
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
    with open(f"{path_to_temp_dir}/{type}/news/{counter}.txt", "a+") as text_file:
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

    for idx, file in enumerate(sorted(os.listdir(f"{path_to_temp_dir}/{type}/news/"), key=lambda x: int(x.split('.')[0]))):

        total = len(os.listdir(f"{path_to_temp_dir}/{type}/news/"))
        print(total)

        input = f"{TEMP_NAME}/{type}/news/{file}"
        output = f"{TEMP_NAME}/{type}/arguments/{idx}"
        subprocess.call([path_to_run, input, output])

        step += 1

        # Write out temporary results
        if step in np.arange(0, total, BATCH_SIZE):
            print(f"WRITE OUT TEMPORARY RESULTS (STEP: {step})")
            print("------------------------------------------------------------------------")
            temp_result = parse_output(type)
            temp_result.to_csv(f"{results_path}/{name}/{type}_batch_{step}.csv", index_label="ID")

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
    
    for dir in sorted(os.listdir(f"{path_to_temp_dir}/{type}/arguments"), key=lambda x: int(x.split('.')[0])):
        with open(f"{path_to_temp_dir}/{type}/arguments/{dir}/OUTPUT.json") as json_file:
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

def run(original: pd.DataFrame, dataset: str, type: str) -> None:
    """
        Args:
            original = DataFrame before MARGOT
            name = string name of dataset. Choices are kaggle, fake_real, liar
    """
    print(f"WRITING {dataset} OUT TO DOCUMENTS")
    print("------------------------------------------------------------------------\n")
    original.apply(lambda x: write_to_doc(x["text"], type=type), axis=1)

    print("CALLING MARGOT")
    print("------------------------------------------------------------------------\n")   
    structure = extract_argumentation(dataset, type)

    print(f"REJOINING: {type}")
    print("------------------------------------------------------------------------\n")   
    result = rejoin_data(original, structure)

    print(f"WRITE FINAL: {type}")
    print("------------------------------------------------------------------------\n")   
    result.to_csv(f"{results_path}/{dataset}/{type}.csv", columns=["ID", "text", "claim", "evidence", "structure", "label"])

if __name__ == "__main__":

    for dataset in os.listdir(path_to_data): 
        if os.path.isdir(f"{path_to_data}/{dataset}") and dataset != ".DS_Store" and dataset == DATASET:

            df_train = pd.read_csv(f"{path_to_data}/{dataset}/train.csv").dropna()
            df_validation = pd.read_csv(f"{path_to_data}/{dataset}/validation.csv").dropna()
            df_test = pd.read_csv(f"{path_to_data}/{dataset}/test.csv").dropna()

            print(f"DATASET: {dataset} - LENGTH TRAIN: {len(df_train)} - LENGTH VALIDATION: {len(df_validation)} - LENGTH TEST: {len(df_test)}")
            print("------------------------------------------------------------------------\n")

            # create_folders(dataset, "train")
            # step, counter = 0, 0
            # print(f"{dataset}: TRAIN")
            # run(df_train, dataset, "train")

            create_folders(dataset, "validation")
            step, counter = 0, 0
            print(f"{dataset}: VALIDATION")
            run(df_validation, dataset, "validation")

            # create_folders(dataset, "test")
            # print(f"{dataset}: TEST")
            # step, counter = 0, 0
            # run(df_test, dataset, "test")