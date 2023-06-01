import os
import pandas as pd
# import sys
import torch
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize


MAX_SECTION = 50

def extract_claim(text):
    sections = divide(text)
    temp = []
    for section in sections:
        print(f"\n# of tokens in section: {len(section.split())}\n")
        prompt = (
            f'Please extract the claim(s) from the following section: <{section}>.'
            f'Do not include any additional information except for the claim(s). '
            f'If no claims can be extracted, please return "NONE" as the value.'
            )
        print(prompt)
        print(f"Tokens: {len(prompt.split()) + len(text.split())}\n")
        claim = generate_text(prompt)
        print(claim)
        print(claim[0]["generated_text"])
        temp.append(claim[0]["generated_text"])
        result = '.'.join(temp)
    return result

def divide(news):
    sentences = sent_tokenize(news)
    sections = []
    current_section = ''
    for sentence in sentences:
        if len(nltk.word_tokenize(current_section + ' ' + sentence)) <= MAX_SECTION:
            current_section += ' ' + sentence
        else:
            sections.append(current_section.strip())
            current_section = sentence
    sections.append(current_section.strip())
    return sections

def extract_evidence(text):
    evidence = generate_text(f"Retrieve the (implicit) main evidences(s) opted by the writer from the following news article: {text}, in case there are more than one, please separate them with '\n'. Do not add any other information than the evidence(s).")
    return evidence[0]["generated_text"]

if __name__ == "__main__":
    #torch.cuda.empty_cache()
    #print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    dir = f"pipeline/argumentation-based"
    for name in os.listdir(f"data"):
        if name != ".DS_Store":
            train = pd.read_csv(f"{dir}/data/{name}/train.csv").dropna()[3:5]
            # test = pd.read_csv(f"{dir}/data/{name}/test.csv").dropna()[3:5]

            print(f"DATASET: {name} - LENGTH TRAIN: {len(train)}")
            print("------------------------------------------------------------------------\n")

            train["claim"] = train.apply(lambda x: extract_claim(x["text"]), axis=1)
            # train["evidence"] = train.apply(lambda x: extract_evidence(x["text"]), axis=1)

            # test["claim"] = test.apply(lambda x: extract_claim(x["text"]), axis=1)
            # test["evidence"] = test.apply(lambda x: extract_evidence(x["text"]), axis=1)

            p = f"{dir}/argumentation structure/dolly"
            if not os.path.exists(p):
                os.makedirs(f"{p}/train")
                # os.makedirs(f"{p}/test")

            print(train.head())
            train.to_csv(f"{p}/train/{name}.csv")
            # test.to_csv(f"{p}/test/{name}.csv")