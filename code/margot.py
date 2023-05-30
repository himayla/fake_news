import pandas as pd
import subprocess
import os
import json
import numpy as np
import shutil

path = "pipeline/argumentation-based/tools/MARGOT/"
tool = "pipeline/argumentation-based/tools/MARGOT/run_margot.sh"

def df_to_doc(clean_text):
    global counter
    with open(f"{path}/temp/news/{counter}.txt", "a+") as text_file:
        text_file.write(clean_text)
    counter += 1
    
def extract_argumentation(name):
    global step
    for idx, file in enumerate(sorted(os.listdir(f"{path}/temp/news"), key=lambda x: int(x.split('.')[0]))):

        total = len(os.listdir(f"{path}/temp/news"))
        print(f"STEP: {step}")
        print("------------------------------------------------------------------------")


        input = f"{name}/temp/news/{file}"
        output = f"{name}/temp/arguments/{idx}"
        subprocess.call([tool, input, output])

        step += 1

        # Write out temporary results
        if step in np.arange(0, total, 50):
            print("WRITE OUT TEMPORARY RESULTS")
            print("------------------------------------------------------------------------")
            res = parse_output()
            temp_result = pd.DataFrame.from_dict(res, orient='index')
            temp_result.to_csv(f"{path}/temp_results/temp_{name}")

    res = parse_output()
    return pd.DataFrame.from_dict(res, orient='index')

def parse_output():
    argument_structures = {}
    for dir in sorted(os.listdir(f"{path}/temp/arguments"), key=lambda x: int(x.split('.')[0])):
        with open(f"{path}/temp/arguments/{dir}/OUTPUT.json") as json_file:

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

            dir = f"{path}/temp"
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(f"{dir}/news", f"{dir}/arguments")

            # Convert news values out to documents
            train.apply(lambda x: df_to_doc(x["text"]), axis=1)
            test.apply(lambda x: df_to_doc(x["text"]), axis=1)

            # Run Margot over the documents
            train_result = extract_argumentation(train, name)
            test_result = extract_argumentation(test, name)

            # Write result data/argumentation_structure
            train_result.to_csv(f"{dir}/argumentation structure/margot/{name}/train.csv")
            test_result.to_csv(f"{dir}/argumentation structure/margot/{name}/test.csv")