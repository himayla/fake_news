import pandas as pd
import subprocess
import os
import json
import numpy as np
import shutil

path_to_data = f"pipeline/argumentation-based/data"
path_to_temp_dir = f"pipeline/argumentation-based/tools/predictor/temp"

path_to_margot = "pipeline/argumentation-based/tools/predictor"
path_to_run = "pipeline/argumentation-based/tools/predictor/run_margot.sh"

results_path = f"pipeline/argumentation-based/argumentation structure/margot"

def write_to_doc(clean_text: str, type: str) -> None:
    """
        Function to convert texts in the DataFrame to temporary text files.
        The function writes out one string to a file, where the filename is the row number.
        Args:
            clean_text = string
            type = string with e.g.: train, test
        Returns:
            Creates a folder with news texts.
    """
    global counter
    with open(f"{path_to_margot}/temp/news/{type}/{counter}.txt", "a+") as text_file:
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

    for idx, file in enumerate(sorted(os.listdir(f"{path_to_margot}/temp/news/{type}"), key=lambda x: int(x.split('.')[0]))):

        total = len(os.listdir(f"{path_to_margot}/temp/news/{type}"))
        print(f"STEP: {step}")
        print("------------------------------------------------------------------------")

        input = f"temp/news/{type}/{file}"
        output = f"temp/arguments/{type}/{idx}"
        subprocess.call([path_to_run, input, output])

        step += 1

        # Write out temporary results
        if step in np.arange(0, total, 100):
            print("WRITE OUT TEMPORARY RESULTS")
            print("------------------------------------------------------------------------")
            temp_result = parse_output(type)
            temp_result.to_csv(f"{results_path}/{name}/{type}.csv")

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
    """
    claim_evidence = {}
    for dir in sorted(os.listdir(f"{path_to_margot}/temp/arguments/{type}"), key=lambda x: int(x.split('.')[0])):
        with open(f"{path_to_margot}/temp/arguments/{type}/{dir}/OUTPUT.json") as json_file:
            doc = json.loads(json_file.read())

            options = {"claim": [], "evidence": []}

            for sent in doc["document"]:

                if sent.get("claim"):
                    options["claim"].append(sent["claim"])

                elif sent.get("evidence"):
                    options["evidence"].append(sent["evidence"])

        claim_evidence[dir] = options
    
    result = pd.DataFrame.from_dict(claim_evidence, orient='index')
 
    return result

if __name__ == "__main__":
    for name in os.listdir(path_to_data):  
        if os.path.isdir(f"{path_to_data}/{name}"):
            train = pd.read_csv(f"{path_to_data}/{name}/train.csv").dropna()
            test = pd.read_csv(f"{path_to_data}/{name}/test.csv").dropna()

            step = 0
            counter = 0

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
    
            # Convert news values out to documents
            train.apply(lambda x: write_to_doc(x["text"], type="train"), axis=1)
            test.apply(lambda x: write_to_doc(x["text"], type="test"), axis=1)

            # Run Margot over the documents
            train_result = extract_argumentation(name, "train")
            test_result = extract_argumentation(name, "test")

            # Write result data/argumentation_structure
            train_result.to_csv(f"{results_path}/{name}/train.csv")
            test_result.to_csv(f"{results_path}/{name}/test.csv")