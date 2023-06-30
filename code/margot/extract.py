# extract.py
#
# Mayla Kersten
#
# Code to extract argumentation from news texts using MARGOT
#
#
import pandas as pd
import subprocess
import os
import json
import shutil

path_to_temp_dir = f"code/margot/predictor/temp"


def create_folders(type: str):
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


def write_to_doc(clean_text: str) -> None:
    """
    Function to convert texts in the DataFrame to temporary text files.
    The function writes out one string to a file, where the filename is the row number (found by counter).
    Args:
        clean_text = str
        type = str with: train, validation, or test
    Returns:
        Creates a folder with news texts.
    """
    global counter
    with open(f"{path_to_temp_dir}/news/{counter}.txt", "a+") as text_file:
        text_file.write(clean_text)
    counter += 1


def extract_argumentation(type: str) -> pd.DataFrame:
    """
    Function calls MARGOT.
    It loads texts from temporary folder and calls MARGOT to extract the claim(s) and evidence(s) to another temporary folder.
    For every 100 texts, it writes out temporary results, this is counted by step.
    Args:
        type = string with e.g.: train, test.
    Returns:
        DataFrame with claim and evidence for each article.
    """
    global step

    for idx, file in enumerate(
        sorted(
            os.listdir(f"{path_to_temp_dir}/news/"), key=lambda x: int(x.split(".")[0])
        )
    ):
        total = len(os.listdir(f"{path_to_temp_dir}/{type}/news/"))
        print(total)

        input = f"temp/{type}/news/{file}"
        output = f"temp/{type}/arguments/{idx}"
        subprocess.call(["code/margot/predictor/run_margot.sh", input, output])

        step += 1

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

    for dir in sorted(
        os.listdir(f"{path_to_temp_dir}/{type}/arguments"),
        key=lambda x: int(x.split(".")[0]),
    ):
        with open(
            f"{path_to_temp_dir}/{type}/arguments/{dir}/OUTPUT.json"
        ) as json_file:
            doc = json.loads(json_file.read())

            options = {"claim": [], "evidence": []}

            for sent in doc["document"]:
                if sent.get("claim"):
                    s = sent["claim"] + '"'
                    options["claim"].append(s.replace(" .", "."))

                elif sent.get("evidence"):
                    s = sent["evidence"]
                    options["evidence"].append(s.replace(" .", "."))

        claim_evidence[dir] = options

    result = pd.DataFrame.from_dict(claim_evidence, orient="index")

    return result


def rejoin_data(original: str, structure: str) -> dict[pd.DataFrame]:
    """
    Function combines annotated results to original.
    Args:

    """
    # Only continue with rows that either contain claim or evidence (not total empty)
    df_structure = structure.loc[
        (structure["claim"] != "[]") | (structure["evidence"] != "[]")
    ]

    merged_df = pd.concat(
        [original.reset_index(drop=True), df_structure.reset_index(drop=True)], axis=1
    )

    # Merge claim & evidence together as "structure"
    merged_df["claim"] = merged_df["claim"].apply(
        lambda x: ", ".join(x) if x else "<unk>"
    )
    merged_df["evidence"] = merged_df["evidence"].apply(
        lambda x: ", ".join(x) if x else "<unk>"
    )
    claims = "claims:" + merged_df["claim"].values
    evidences = "evidences:" + merged_df["evidence"].values

    merged_df["structure"] = claims + evidences
    merged_df.dropna(inplace=True)

    return merged_df


def extract(dataset: pd.DataFrame, type: str) -> None:
    """
    Args:
        original = DataFrame before MARGOT
        name = string name of dataset. Choices are kaggle, fake_real, liar
    """
    print(f"WRITING {type} OUT TO DOCUMENTS")
    print("------------------------------------------------------------------------\n")
    dataset.apply(lambda x: write_to_doc(x["text"], type=type), axis=1)

    print("CALLING MARGOT")
    print("------------------------------------------------------------------------\n")
    structure = extract_argumentation(dataset, type)

    print(f"REJOINING: {type}")
    print("------------------------------------------------------------------------\n")
    result = rejoin_data(dataset, structure)

    print(f"WRITE FINAL: {type}")
    print("------------------------------------------------------------------------\n")

    result.to_csv(
        f"data/margot/{type}.csv",
        columns=["ID", "text", "claim", "evidence", "structure", "label"],
    )


def load_data(path: str) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = []
    for i in files:
        df = pd.read_csv(f"{path}/{i}.csv", index_col="ID")
        data.append(df)

    return df


if __name__ == "__main__":
    data = load_data()
    files = ["train", "validation", "test"]

    for i in range(len(data)):
        create_folders(data[i], files[i])
        step, counter = 0, 0
        extract(data[i], files[i])
