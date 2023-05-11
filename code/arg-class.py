import argparse
import pandas as pd
import subprocess
import loader
import os
import re
import string
import json
import contractions

def df_to_doc(clean_text):
    # TODO: Fix cleaning issues
    global counter
    with open(f"temp/news/{counter}.txt", "a+") as text_file:
        expanded_text = []
        for word in clean_text.split():
            expanded_text.append(contractions.fix(word))  
        clean_text = ' '.join(expanded_text)
        clean_text = re.sub(r'[``“”’,"\'-]', "", clean_text) # Remove weird quotation things
        clean_text = re.sub(r'\n', " ", clean_text) # Remove weird quotation things
        text_file.write(clean_text)
    counter += 1

def extract_argumentation():
    tool = "AM/MARGOT/run_margot.sh" # Default
    for idx, file in enumerate(sorted(os.listdir("temp/news"))):
        input = f"../../temp/news/{file}"
        output = f"../../temp/arguments/{idx}"
        subprocess.call([tool, input, output])

    res = parse_output()

    res = {k: [v] for k, v in res.items()}

    return pd.DataFrame.from_dict(res, orient='index')



def parse_output():
    argument_structures = {}
    for dir in sorted(os.listdir("temp/arguments")):
        with open(f"temp/arguments/{dir}/OUTPUT.json") as json_file:
            doc = json.loads(json_file.read())
            args = []

            for sent in doc["document"]:

                if sent.get("claim"):
                    args.append((sent["claim"], "claim"))

                elif sent.get("evidence"):
                    args.append((sent["evidence"], "evidence"))
        

        argument_structures[dir] = args
    
    return argument_structures

if __name__ == "__main__":
    counter = 0 

    # TODO: Implement command-line arguments so that different AM-tools can be used
    # parse_commands()

    data = loader.load_data("data")
    
    for name, df in data.items():
        print(f"Data: {name}, {len(df)}")
    
        # Convert news values out to documents
        df.apply(lambda x: df_to_doc(x["text"]), axis=1)

        # Run Margot over the documents
        result = extract_argumentation()

        # Write result data/argumentation_structure
        result.to_csv(f"am/output/{name}.csv")

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