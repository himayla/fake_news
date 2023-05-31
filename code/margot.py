import pandas as pd
import subprocess
import os
import json
import numpy as np
import shutil

path = "pipeline/argumentation-based/tools/predictor"
tool = "pipeline/argumentation-based/tools/predictor/run_margot.sh"

def df_to_doc(clean_text, type):
    global counter
    print(f"Writing to {path}/temp/news/{type}")
    with open(f"{path}/temp/news/{type}/{counter}.txt", "a+") as text_file:
        text_file.write(clean_text)
    counter += 1
    
def extract_argumentation(name, type):
    global step
    for idx, file in enumerate(sorted(os.listdir(f"{path}/temp/news/{type}"), key=lambda x: int(x.split('.')[0]))):
        print(idx, file)

        total = len(os.listdir(f"{path}/temp/news/{type}"))
        print(f"STEP: {step}")
        print("------------------------------------------------------------------------")

        input = f"temp/news/{type}/{file}"
        output = f"temp/arguments/{type}/{idx}"
        subprocess.call([tool, input, output])

        step += 1

        # Write out temporary results
        if step in np.arange(0, total, 50):
            print("WRITE OUT TEMPORARY RESULTS")
            print("------------------------------------------------------------------------")
            res = parse_output(type)
            temp_result = pd.DataFrame.from_dict(res, orient='index')
            if os.path.exists(f"{path}/argumentation structure/margot/temp_{name}"):
                os.makedirs(f"{path}/argumentation structure/margot/temp_{name}")
            temp_result.to_csv(f"{path}/argumentation structure/margot/temp_{name}/{type}.csv")

    res = parse_output(type)
    return pd.DataFrame.from_dict(res, orient='index')

def parse_output(type):
    argument_structures = {}
    for dir in sorted(os.listdir(f"{path}/temp/arguments/{type}"), key=lambda x: int(x.split('.')[0])):
        with open(f"{path}/temp/arguments/{type}/{dir}/OUTPUT.json") as json_file:

            doc = json.loads(json_file.read())
            args = {"claim": [], "evidence": []}

            for sent in doc["document"]:

                if sent.get("claim"):
                    args["claim"].append(sent["claim"])

                elif sent.get("evidence"):
                    args["evidence"].append(sent["evidence"])

        argument_structures[dir] = args
    
    return argument_structures

if __name__ == "__main__":
    dir = f"pipeline/argumentation-based/data"
    for name in os.listdir(dir):  
        if os.path.isdir(f"{dir}/{name}"):
            train = pd.read_csv(f"{dir}/{name}/train.csv").dropna()
            test = pd.read_csv(f"{dir}/{name}/test.csv").dropna()

            step = 0
            counter = 0

            path_to_temp_dir = f"pipeline/argumentation-based/tools/predictor/temp"
            if os.path.exists(path_to_temp_dir):
                shutil.rmtree(path_to_temp_dir)
            os.makedirs(f"{path_to_temp_dir}/news/train")
            os.makedirs(f"{path_to_temp_dir}/news/test")

            os.makedirs(f"{path_to_temp_dir}/arguments/train")
            os.makedirs(f"{path_to_temp_dir}/arguments/test")

            # Convert news values out to documents
            train.apply(lambda x: df_to_doc(x["text"], type="train"), axis=1)
            test.apply(lambda x: df_to_doc(x["text"], type="test"), axis=1)

            # # Run Margot over the documents
            train_result = extract_argumentation(name, "train")
            test_result = extract_argumentation(name, "train")

            os.makedirs(f"{path}/argumentation structure/margot/train")
            os.makedirs(f"{path}/argumentation structure/margot/test")

            path_to_train = f"{os.getcwd()}/pipeline/argumentation-based/argumentation structure/margot/train"
            path_to_test = f"{os.getcwd()}/pipeline/argumentation-based/argumentation structure/margot/test"
            if not os.path.exists(path_to_temp_dir):
                os.makedirs(path_to_train)
                os.makedirs(path_to_test)

            # Write result data/argumentation_structure
            train_result.to_csv(f"{path_to_train}/{name}.csv")
            test_result.to_csv(f"{path_to_test}/{name}.csv")