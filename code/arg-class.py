import argparse
import pandas as pd
import subprocess
import loader
import os
import json
from tqdm import tqdm
import numpy as np
import shutil


def df_to_doc(clean_text):
    global counter
    with open(f"am/MARGOT/version_1/temp/news/{counter}.txt", "a+") as text_file:
        text_file.write(clean_text)
    counter += 1
    
def extract_argumentation():
    global step
    tool = "am/MARGOT/run_margot.sh" # Default AM tool
    for idx, file in enumerate(sorted(os.listdir("am/MARGOT/version_1/temp/news"), key=lambda x: int(x.split('.')[0]))):

        print(idx, file)
        total = len(f"temp/news/{file}")
        print("------------------------------------")

        input = f"version_1/temp/news/{file}"
        output = f"version_1/temp/arguments/{idx}"
        subprocess.call([tool, input, output])

        step += 1
        print(f"At step:{step}")
        print()

        # Write out temporary results
        if step in np.arange(0, total, 5):
            res = parse_output()
            temp_result = pd.DataFrame.from_dict(res, orient='index')
            temp_result.to_csv(f"am/MARGOT/version_1/temp_results/temp_{name}.csv")
            temp_result.to_csv(f"am/MARGOT/version_1/temp_results/temp_{name}.xlsx")            

        print("------------------------------------")

    res = parse_output()

    #res = {k: [v] for k, v in res.items()}

    return pd.DataFrame.from_dict(res, orient='index')


def parse_output():
    argument_structures = {}
    for dir in sorted(os.listdir("am/MARGOT/version_1/temp/arguments"), key=lambda x: int(x.split('.')[0])):
        print(dir)
        with open(f"am/MARGOT/version_1/temp/arguments/{dir}/OUTPUT.json") as json_file:

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
    counter = 0
    step = 0

    tqdm.pandas()

    data = loader.load_data(arg=True)
    
    for name, df in data.items():
        print(f"Data: {name}, {len(df)}")

        df.dropna(inplace=True)

        dir = "am/MARGOT/version_1/temp"
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(f"{dir}/news")
        os.makedirs(f"{dir}/arguments")

        # Convert news values out to documents
        df.progress_apply(lambda x: df_to_doc(x["text"]), axis=1)

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