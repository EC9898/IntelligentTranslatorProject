"""
Baseline: translate the test set with MarianMT, no refinement.

Reads data/test_set.csv and writes data/baseline_results.csv with the raw
translation plus an empty baseline_rating column (good / awkward / wrong).

Careful: re-running this overwrites the ratings in baseline_results.csv.

Needs: pip install -r requirements.txt
"""

import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

MODELS = {
    "zh": "Helsinki-NLP/opus-mt-zh-en",  # Mandarin -> English
    "en": "Helsinki-NLP/opus-mt-en-zh",  # English -> Mandarin
}

loaded = {}
for src, name in MODELS.items():
    print(f"Loading {name}...")
    loaded[src] = (MarianTokenizer.from_pretrained(name),
                   MarianMTModel.from_pretrained(name))


def translate(text: str, src: str) -> str:
    tokenizer, model = loaded[src]
    if src == "en":
        # en-zh model covers several Chinese variants, this token picks Simplified Mandarin
        text = ">>cmn_Hans<< " + text
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    output_ids = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


df = pd.read_csv("data/test_set.csv", encoding="utf-8-sig", keep_default_na=False)

raw = []
for _, row in df.iterrows():
    result = translate(row["source"], row["source_lang"])
    raw.append(result)
    print(f"[{row['id']}] {row['source']}\n    -> {result}")

df["raw_translation"] = raw
df["baseline_rating"] = ""   # rated by hand afterwards
df.to_csv("data/baseline_results.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved {len(df)} rows to data/baseline_results.csv")
