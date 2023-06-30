# extract.py
#
# Mayla Kersten
#
# Code to extract argumentation from news texts using Dolly 2.0
#
#
from datasets import Dataset as HF_Dataset
import re
from instruct_pipeline import InstructionTextGenerationPipeline
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SECTION = 300
BATCH_SIZE = 100


class NewsDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        return example

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts argumentation from a batch of texts
        """
        texts = batch["text"]

        sections = [divide(text) for text in texts]

        claims, evidences = [], []
        for section_list in sections:
            section_claims, section_evidences = [], []
            for section in section_list:
                print(f"LENGTH SECTION: {len(section.split())}")
                print(
                    "-------------------------------------------------------------------\n"
                )
                claim_prompt = f"Please state the main claim made by the author in the following section of a news article: {section}"
                evidence_prompt = f"Please state the main evidence provided by the author in the following section of a news article: {section}"

                claim_responses = text_generator(claim_prompt)
                evidence_responses = text_generator(evidence_prompt)
                claim_response = re.sub(r"\"\"\"", '"', claim_responses[0]["response"])
                evidence_response = re.sub(
                    r"\"\"\"", '"', evidence_responses[0]["response"]
                )

                section_claims.append(claim_response)
                section_evidences.append(evidence_response)

            claims.append(section_claims)
            evidences.append(section_evidences)

        batch["claim"] = [", ".join(i for i in cl) for cl in claims]
        batch["evidence"] = [", ".join(i for i in ev) for ev in evidences]

        res_claims = ["claims:" + ", ".join(i for i in cl) for cl in claims]
        res_evidences = ["evidences:" + ", ".join(i for i in ev) for ev in evidences]

        batch["structure"] = [
            " ".join(str(i) for i in items) for items in zip(res_claims, res_evidences)
        ]

        return batch

def divide(news: str) -> list:
    """
    Divides text into paragraphs
    """
    sentences = sent_tokenize(news)
    sections = []
    current_section = ""
    for sentence in sentences:
        if len(word_tokenize(current_section + " " + sentence)) <= MAX_SECTION:
            current_section += " " + sentence
        else:
            sections.append(current_section.strip())
            current_section = sentence
    sections.append(current_section.strip())
    return sections


def load_dolly() -> InstructionTextGenerationPipeline:
    """
    Loads Dolly 12b from Huggingface
    """
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
    model = AutoModelForCausalLM.from_pretrained(
        "databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16
    ).to(device)
    text_generator = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    return text_generator


def load_data(path: str) -> Tuple[Dataset, Dataset]:
    
    data = []
    for i in ["train", "validation", "test"]:
        df = pd.read_csv(f"{path}/{i}.csv", index_col="ID")
        data.append(df)

    return df



def extract(dataset):
    """
    Extracts argumentation from texts
    """
    datas = NewsDataset(dataset)
    dataloader = DataLoader(datas, batch_size=BATCH_SIZE)

    processed = []
    for i, batch in enumerate(dataloader):
        processed_batch = datas.process_batch(batch)
        batch_df = pd.DataFrame(processed_batch)
        batch_df.set_index("ID", inplace=True)

        processed.append(batch_df)

    result = pd.concat(processed)

    return result


if __name__ == "__main__":
    text_generator = load_dolly()
    files = ["train", "validation", "test"]
    data = load_data()

    for i in range(len(data)):
        result = extract(data[0], files[0])
        result.to_csv(
            f"dolly/{i}.csv",
            columns=["text", "claim", "evidence", "structure", "label"],
            index_label="ID",
        )
