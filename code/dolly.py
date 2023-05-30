import pandas as pd
import sys; print(sys.path)
import torch; print(torch.__version__)
from transformers import pipeline
import os

def extract_claim(text):
    claim = generate_text(f"Retrieve the (implicit) claim(s) opted by the writer from the following news article: {text}, in case there are more than one, please separate them with '\n'. Do not add any other information than the claim(s).")
    return claim

def extract_evidence(text):
    evidence = generate_text(f"Retrieve the (implicit) evidences(s) opted by the writer from the following news article: {text}, in case there are more than one, please separate them with '\n'. Do not add any other information than the evidence(s).")
    return evidence

if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    dir = f"pipeline/argumentation-based/data"
    for name in os.listdir(dir):
        train = pd.read_csv(f"{dir}/{name}/train.csv").dropna()
        test = pd.read_csv(f"{dir}/{name}/test.csv").dropna()

        print(f"DATASET: {name} - LENGTH TRAIN: {len(train)}")
        print("------------------------------------------------------------------------\n")

        train["claim"] = train.apply(lambda x: extract_claim(x["text"]), axis=1)
        train["evidence"] = train.apply(lambda x: extract_evidence(x["text"]), axis=1)

        print(train.head())
        train.to_csv(f"{dir}/argumentation structure/dolly/{name}/train.csv")
        test.to_csv(f"{dir}/argumentation structure/dolly/{name}/test.csv")