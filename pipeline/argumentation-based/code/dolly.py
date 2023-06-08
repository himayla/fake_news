import os
import pandas as pd
# import sys
import torch
# from transformers import pipeline
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


MAX_SECTION = 300

# def extract_claim(text):
#     global step
#     sections = divide(text)
#     print(f"# OF {sections}")
#     claims, evidences = []
#     for section in sections:
#         claim_prompt = (f"Provide the main claim from the section of a news article below.\n{section}")
#         evidence_prompt = (f"Provide the evidence from the section of a news article below.\n{section}")

#         print(f"Tokens: {len(section.split()) + len(text.split())}\n")
#         claim = generate_text(claim_prompt)
#         evidence = evidence_prompt
#         temp.append(claim[0]["response"])

#         result = ', '.join(temp)
#         step += 1

#     return result

# def extract_evidence(text):
#     global step
#     sections = divide(text)
#     temp = []
#     for section in sections:

#         print(f"Tokens: {len(section.split()) + len(text.split())}\n")
#         claim = generate_text(prompt)
#         temp.append(claim[0]["response"])

#         result = ', '.join(temp)
#     step += 1
#     return result

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

def extract_arg(example):
    global step

    text = example["text"]

    sections = divide(text)
    print(f"# OF {len(sections)}")

    claims, evidences = [], []

    for section in sections:
        claim_prompt = (f"Provide the main claim from the section of a news article below.\n{section}")
        evidence_prompt = (f"Provide the main evidence from the section of a news article below.\n{section}")

        print(f"TOKENS: {len(section.split()) + len(text.split())}\n")

        claim = generate_text(claim_prompt)
        evidence = generate_text(evidence_prompt)

        claims.append(claim[0]["response"])
        evidences.append(evidence[0]["response"])
        step += 1

        # Write out temporary results
        # if step in np.arange(0, 10000, 25):
        #     print("WRITE OUT TEMPORARY RESULTS")
        #     print("------------------------------------------------------------------------")
        #     example["claim"] = ', '.join(claims)
        #     example["evidence"] = ', '.join(evidences)
        #     example.to_csv(f"{p}/{name}/{type}.csv", columns=["text", "claim", "evidence", "label"])

    example["claim"] = ', '.join(claims)
    example["evidence"] = ', '.join(evidences)

    return example

if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16).to("cuda")
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
    
    dir = f"pipeline/argumentation-based"
    for name in os.listdir(f"data"):
        step = 0
        if name != ".DS_Store":
            train = pd.read_csv(f"{dir}/data/{name}/train.csv").dropna()
            test = pd.read_csv(f"{dir}/data/{name}/test.csv").dropna()

            print(f"DATASET: {name} - LENGTH TRAIN: {len(train)}")
            print("------------------------------------------------------------------------\n")

            p = f"{dir}/argumentation structure/dolly"
            if not os.path.exists(f"{p}/{name}"):
                os.makedirs(f"{p}/{name}")

            train = train.apply(lambda x: extract_arg(x), axis=1)
            test = test.apply(lambda x: extract_arg(x), axis=1)
    
            train.set_index("ID", inplace=True)
            test.set_index("ID", inplace=True)

            print(train.head())
            print(test.head())

            train.to_csv(f"{p}/{name}/train.csv", columns=["text", "claim", "evidence", "label"])
            test.to_csv(f"{p}/{name}/test.csv", columns=["text", "claim", "evidence", "label"])