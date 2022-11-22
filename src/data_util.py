
import json

import pandas as pd
from transformers import BertTokenizer, CamembertTokenizerFast


def fetch_dataset(filepath: str):
    items = []
    with open(filepath, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            js_obj = json.loads(line)
            items.append(js_obj)
            line = f.readline()

    return pd.DataFrame(items)

def transform_tokenize(df: pd.DataFrame):
    tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

    body = df["review_body"]
    body_max = max(body.str.len())
    print(body_max)

    title = df["review_title"]
    aaa = tokenizer(body.tolist(),
              padding='max_length', max_length=512, truncation=True,
              return_tensors="pt")
    print(aaa)
    print(body)
    body_token = tokenizer.tokenize(body[0])


    return body_token, title_token
