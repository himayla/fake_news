import argparse
import pandas as pd
import subprocess
import loader
import os
import json
import numpy as np
import shutil
import write_out

job_name = "kaggle_2000"
path = "am/MARGOT/kaggle_2000"
tool = "am/MARGOT/run_margot.sh" # Default AM tool

def df_to_doc(clean_text):
    global counter
    with open(f"{path}/temp/news/{counter}.txt", "a+") as text_file:
        text_file.write(clean_text)
    counter += 1
    
def extract_argumentation():
    global step
    for idx, file in enumerate(sorted(os.listdir(f"{path}/temp/news"), key=lambda x: int(x.split('.')[0]))):

        total = len(os.listdir(f"{path}/temp/news"))
        print(f"STEP: {step}")
        print("------------------------------------------------------------------------")


        input = f"{job_name}/temp/news/{file}"
        output = f"{job_name}/temp/arguments/{idx}"
        subprocess.call([tool, input, output])

        step += 1

        # Write out temporary results
        if step in np.arange(0, total, 50):
            print("WRITE OUT TEMPORARY RESULTS")
            print("------------------------------------------------------------------------")
            res = parse_output()
            temp_result = pd.DataFrame.from_dict(res, orient='index')
            write_out.write_data(temp_result, f"{path}/temp_results/temp_{name}")

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
    # TODO: Implement command-line arguments so that different AM-tools can be used
    # parse_commands()

    data = loader.load_data_arg("data/original", "data/clean/arg")
    
    for name, df in data.items():
        if name == "kaggle": ##
            step = 0
            counter = 0

            df.dropna(inplace=True)

            dir = f"{path}/temp"
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(f"{dir}/news")
            os.makedirs(f"{dir}/arguments")

            # Convert news values out to documents
            df.apply(lambda x: df_to_doc(x["text"]), axis=1)

            # Run Margot over the documents
            result = extract_argumentation()

        # Write result data/argumentation_structure
        result.to_csv(f"am/MARGOT/version_1/fin_{name}.csv")
        result.to_csv(f"am/MARGOT/version_1/fin_{name}.xlsx")

    # TODO: Train on the result

# def parse_commands():
#     parser = argparse.ArgumentParser(prog='AM')

#     parser.add_argument("--tool", help="Tool to use")

#     try:
#         args = parser.parse_args()
#         if args.tool == tools[0]:
#             pass
#         else:
#             path = ""
#     except:
#         pass