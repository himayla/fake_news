import os
import pandas as pd
import re

PATH = "pipeline/argumentation-based/argumentation structure/dolly/kaggle_4000"

def combine_batches(batches, name, replace=False) -> None:
    """
        Combine different batches to one DataFrame and writes this out
        Used for e.g., in case Slurm cancels job
        If replace, delete batches
    """
    # Sort batches from to preserve original order (not necessary, but nice to have)
    sorted_files = sorted(batches, key=lambda x: int(re.findall(r'\d+', x)[0]))


    batched = []
    for f in sorted_files[:2]:
        batched.append(pd.read_csv(f"{PATH}/{f}"))

    res = pd.concat(batched, ignore_index=True)
    res.set_index("ID", inplace=True)

    res.to_csv(f"{PATH}/{name}.csv")

if __name__ == "__main__":

    batches_train, batches_test = [], []

    for f in os.listdir(PATH):
        if f.startswith("train_"):
           batches_train.append(f)
        else:
            batches_test.append(f)

    if len(batches_train) > 1:
        combine_batches(batches_train, "train")

    if len(batches_test) > 1:
        combine_batches(batches_test, "test")

