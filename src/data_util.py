
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

