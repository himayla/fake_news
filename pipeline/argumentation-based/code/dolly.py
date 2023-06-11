from datetime import datetime
import os
import pandas as pd
import torch
import nltk
from datasets import Dataset as HF_Dataset
from nltk.tokenize import sent_tokenize
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

DATASET = "sample"
BATCH_SIZE = 3
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
        global step

        current_time = datetime.now()
        print(f"PROCESSING BATCH - STEP {step} - {current_time.hour}:{current_time.minute}")
        print("-------------------------------------------------------------------\n")
        texts = batch["text"]

        sections = [divide(text) for text in texts]

        claims, evidences = [], []
        for i, section_list in enumerate(sections):
            section_claims, section_evidences = [], []
            
            for s, section in enumerate(section_list):
                claim_prompt = f"Without providing any interpretation, please state the main claim made by the author in the following section of a news article:\n{section}"
                evidence_prompt = f"Without providing any interpretation, please state the main evidence provided by the author in the following section of a news article:\n{section}"

                claim_responses = generate_text(claim_prompt)
                evidence_responses = generate_text(evidence_prompt)

                section_claims.append([response["response"] for response in claim_responses])
                section_evidences.append([response["response"] for response in evidence_responses])

            claims.append(sum(section_claims, []))
            evidences.append(sum(section_evidences, []))

        batch["claim"] = ', '.join(sum(section_claims, []))
        batch["evidence"] = ', '.join(sum(section_claims, []))

        return batch


def divide(news):
    print("DIVIDE NEWS ARTICLE")
    print("-------------------------------------------------------------------\n")
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

def run(data, type):
    datas = NewsDataset(data)
    dataloader = DataLoader(datas, batch_size=BATCH_SIZE)
    
    processed = []
    for i, batch in enumerate(dataloader):
        processed_batch = datas.process_batch(batch)
        batch_df = pd.DataFrame(processed_batch)
        batch_df.set_index("ID", inplace=True)

        # Save processed batch to CSV
        print("WRITE OUT TEMPORARY FILE")
        print("-------------------------------------------------------------------\n")

        # batch_df["ID"] = batch_df["ID"].astype(int)
        processed.append(batch_df)
        batch_df.to_csv(f"{p}/{name}/{type}_{i}_batch.csv", columns=["text", "claim", "evidence", "label"])

    result = pd.concat(processed) 
    result.set_index("ID", inplace=True)

    result.to_csv(f"{p}/{name}/{type}.csv", columns=["claim", "evidence", "label"]) #  REMOVED TEXT FOR CHECK

    return result

if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16).to("cuda")
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    dir = f"pipeline/argumentation-based"
    for name in os.listdir(f"{dir}/data"):

        step = 0
        if name != ".DS_Store" and name == DATASET:
            print(f"DATASET: {name}")
            print("--------------------------------------------------------------------\n")
            
            train = pd.read_csv(f"{dir}/data/{name}/train.csv").dropna()
            test = pd.read_csv(f"{dir}/data/{name}/test.csv").dropna()


            print(f"LENGTH TRAIN: {len(train)} - LENGTH TEST {len(test)}")
            print("------------------------------------------------------------------------\n")
            
            train = HF_Dataset.from_pandas(train, preserve_index=True).class_encode_column("label")
            test = HF_Dataset.from_pandas(test, preserve_index=True).class_encode_column("label")

            p = f"{dir}/argumentation structure/dolly"
            if not os.path.exists(f"{p}/{name}"):
                os.makedirs(f"{p}/{name}")

            run(train, "train")
            run(test, "test")