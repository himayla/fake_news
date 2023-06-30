from datetime import datetime
import os
import pandas as pd
import torch
import nltk
from datasets import Dataset as HF_Dataset
from nltk.tokenize import sent_tokenize; 
import nltk#; nltk.download('punkt')
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import re

DATASET = "fake_real_1000"
BATCH_SIZE = 100
MAX_SECTION = 400

class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return example

    def process_batch(self, batch):
        # Process the example
        global step

        current_time = datetime.now()
        print(f"PROCESSING BATCH - STEP {step} - {current_time.hour}:{current_time.minute}")
        print("-------------------------------------------------------------------\n")
        texts = batch["text"]

        sections = [divide(text) for text in texts]

        claims, evidences = [], []
        for section_list in sections:

            section_claims, section_evidences = [], []
            for section in section_list:

                print(f"LENGTH SECTION: {len(section.split())}")
                print("-------------------------------------------------------------------\n")
                claim_prompt = f"Please state the main claim made by the author in the following section of a news article: {section}"
                evidence_prompt = f"Please state the main evidence provided by the author in the following section of a news article: {section}"

                print(claim_prompt)

                # try:
                claim_responses = generate_text(claim_prompt)
                evidence_responses = generate_text(evidence_prompt)
                claim_response = re.sub(r'\"\"\"', '"', claim_responses[0]["response"])
                evidence_response = re.sub(r'\"\"\"', '"', evidence_responses[0]["response"])
                # except RuntimeError:
                #     print(section)
                #     print(batch["ID"])
                #     claim_response = "<unk>"
                #     evidence_response = "<unk>"
            
                section_claims.append(claim_response)
                section_evidences.append(evidence_response)

            claims.append(section_claims)
            evidences.append(section_evidences)

        batch["claim"] = [', '.join(i for i in cl) for cl in claims]
        batch["evidence"] = [', '.join(i for i in ev) for ev in evidences]
        res_claims = ['claims:' + ', '.join(i for i in cl) for cl in claims]
        res_evidences = ['evidences:' + ', '.join(i for i in ev) for ev in evidences]
        batch["structure"] =  [' '.join(str(i) for i in items) for items in zip(res_claims, res_evidences)]

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
    global step
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

        processed.append(batch_df)
        batch_df.to_csv(f"{p}/{name}/{type}_{i}_batch.csv", columns=["text", "claim", "evidence", "structure", "label"])

        step += 1

    result  = pd.concat(processed)
    print(result.head()) 

    result.to_csv(f"{p}/{name}/{type}.csv", columns=["text", "claim", "evidence","structure", "label"])

    return result

if __name__ == "__main__":

    # torch.cuda.empty_cache()
    # print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    # tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
    # model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16).to("cuda")
    # generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    # dolly_example = pd.DataFrame(
    #     index=[0],
    #     data={"text":"Honduras bans citizens from owning guns and has the highest homicide rate in the entire world. Switzerland, with a similar population, requires citizens to own guns and has the lowest homicide rate in the entire world. Barack Obama and Hillary Clinton are negotiating with the United Nations about doing a treaty that will ban the use of firearms.", "label": "FAKE"})
    # # example = HF_Dataset.from_pandas(dolly_example, preserve_index=True).class_encode_column("label")

    # claim_prompt = f"Please state the main claim made by the author in the following section of a news article: {dolly_example['text']}"
    # evidence_prompt = f"Please state the main evidence provided by the author in the following section of a news article: {dolly_example['text']}"

    # print(claim_prompt)

    # # try:
    # claim_responses = generate_text(claim_prompt)
    # evidence_responses = generate_text(evidence_prompt)
    # claim_response = re.sub(r'\"\"\"', '"', claim_responses[0]["response"])
    # evidence_response = re.sub(r'\"\"\"', '"', evidence_responses[0]["response"])

    # print(claim_response)
    # print(evidence_response)
# dir = f"pipeline/argumentation-based"
    for name in os.listdir(f"{dir}/data"):

        step = 0
        if name != ".DS_Store" and name == DATASET:
            print(f"DATASET: {name}")
            print("--------------------------------------------------------------------\n")
            
            train = pd.read_csv(f"{dir}/data/{name}/train.csv").dropna()
            val = pd.read_csv(f"{dir}/data/{name}/validation.csv").dropna()
            test = pd.read_csv(f"{dir}/data/{name}/test.csv").dropna()[:50]

            print(f"LENGTH TRAIN: {len(train)} - LENGTH VAL: {len(val)} - LENGTH TEST {len(test)}")
            print("------------------------------------------------------------------------\n")
            
            train = HF_Dataset.from_pandas(train, preserve_index=True).class_encode_column("label")
            val = HF_Dataset.from_pandas(val, preserve_index=True).class_encode_column("label")
            test = HF_Dataset.from_pandas(test, preserve_index=True).class_encode_column("label")

            p = f"{dir}/argumentation structure/dolly"
            if not os.path.exists(f"{p}/{name}"):
                os.makedirs(f"{p}/{name}")

            run(train, "train")
            run(val, "validation")
            run(test, "test")