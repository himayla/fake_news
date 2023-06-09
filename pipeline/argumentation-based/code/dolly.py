import os
import pandas as pd
import torch
import numpy as np
import nltk
from datasets import Dataset as HF_Dataset
from nltk.tokenize import sent_tokenize
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

MAX_SECTION = 300

class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return example  # Convert to dictionary

    def process_batch(self, batch):
        # Process the example
        print("PROCESSING BATCH")
        texts = batch["text"]
        sections = [divide(text) for text in texts]
        claims, evidences = [], []

        for section_list in sections:
            section_claims, section_evidences = [], []
            
            for section in section_list:
                claim_prompt = f"Provide the main claim from the section of a news article below.\n{section}"
                evidence_prompt = f"Provide the main evidence from the section of a news article below.\n{section}"
                
                claim_responses = generate_text(claim_prompt)
                evidence_responses = generate_text(evidence_prompt)
                
                section_claims.extend([response["response"] for response in claim_responses])
                section_evidences.extend([response["response"] for response in evidence_responses])
            
            claims.append(', '.join(section_claims))
            evidences.append(', '.join(section_evidences))

        batch["claim"] = claims
        batch["evidence"] = evidences

        return batch


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


if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16).to("cuda")
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    dir = f"pipeline/argumentation-based"
    for name in os.listdir(f"data"):
        step = 0
        if name != ".DS_Store" and name != "kaggle":
            train = pd.read_csv(f"{dir}/data/{name}/train.csv").dropna()
            # print(train)
            test = pd.read_csv(f"{dir}/data/{name}/test.csv").dropna()

            print(f"DATASET: {name} - LENGTH TRAIN: {len(train)}")
            print("------------------------------------------------------------------------\n")
            
            train = HF_Dataset.from_pandas(train).class_encode_column("label")
            test = HF_Dataset.from_pandas(test).class_encode_column("label")

            train_dataset = NewsDataset(train)
            train_dataloader = DataLoader(train_dataset, batch_size=32)

            test_dataset = NewsDataset(test)
            test_dataloader = DataLoader(test_dataset, batch_size=32)

            p = f"{dir}/argumentation structure/dolly"
            if not os.path.exists(f"{p}/{name}"):
                os.makedirs(f"{p}/{name}")

            processed_train = []
            for i, batch in enumerate(train_dataloader):
                processed_batch = train_dataset.process_batch(batch)
                processed_train.append(processed_batch)

                # Save processed batch to CSV
                print("Write out temporary file")
                batch_df = pd.DataFrame(processed_batch)
                batch_df.set_index("ID", inplace=True)
                batch_df.to_csv(f"{p}/{name}/train_batch_{i}.csv", columns=["text", "claim", "evidence", "label"])

            train = pd.concat(processed_train, ignore_index=True)

            processed_test = []
            for i, batch in enumerate(test_dataloader):
                processed_batch = train_dataset.process_batch(batch)
                processed_test.append(processed_batch)

                # Save processed batch to CSV
                print("Write out temporary file")

                batch_df = pd.DataFrame(processed_batch)
                batch_df.set_index("ID", inplace=True)
                batch_df.to_csv(f"{p}/{name}/test_batch_{i}.csv", columns=["text", "claim", "evidence", "label"])

            test = pd.concat(processed_test, ignore_index=True)

            train.set_index("ID", inplace=True)
            test.set_index("ID", inplace=True)

            print(train.head())
            print(test.head())

            train.to_csv(f"{p}/{name}/train.csv", columns=["text", "claim", "evidence", "label"])
            test.to_csv(f"{p}/{name}/test.csv", columns=["text", "claim", "evidence", "label"])