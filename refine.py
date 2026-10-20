"""
Refinement: a local Qwen model (through Ollama) post-edits the MarianMT output.

Reads data/baseline_results.csv and writes data/refined_<model>.csv with the
refined text, whether the model changed anything, and its reason.

Setup:
    1. Install Ollama from https://ollama.com (runs locally, so not in Colab)
    2. ollama pull qwen3.5:9b   (qwen3.5:4b or qwen3.5:2b for slower machines)
    3. pip install -r requirements.txt

Usage:
    python refine.py qwen3.5:9b
"""

import json
import sys
import pandas as pd
import ollama

if len(sys.argv) != 2:
    sys.exit("Usage: python refine.py <model>   (e.g. python refine.py qwen3.5:2b)")

MODEL = sys.argv[1]  # has to match a name from `ollama list`
# separate output file per model so runs don't overwrite each other
# (windows doesn't allow ":" in filenames, so qwen3.5:2b -> refined_qwen3.5-2b.csv)
OUT_PATH = f"data/refined_{MODEL.replace(':', '-')}.csv"

LANG_NAMES = {"zh": "Mandarin Chinese", "en": "English"}

# The model should post-edit, not rewrite: fix what's wrong, leave correct
# translations alone, never add information.
SYSTEM_PROMPT = """You are a translation post-editor. You receive a source \
sentence and a machine translation of it. Fix errors in the translation while \
preserving the speaker's meaning and tone.

Rules:
- Fix mistranslations, slang or idioms translated literally, and unnatural grammar.
- If the source leaves out a subject (common in Chinese), work out who it \
refers to from the sentence and use the correct pronoun.
- Match the speaker's register: casual stays casual. Slang becomes natural \
slang or casual phrasing in the target language.
- If the translation is already accurate and natural, return it unchanged.
- Never add information that is not in the source.

Respond with only JSON: {"refined": string, "changed": boolean, "reason": string}"""

# Few-shot examples made by Claude. Never use test set sentences here (data leakage).
# The last one is already correct, so the model sees that "no change" is allowed.
FEW_SHOT = [
    ("zh", "en", "我给她打电话，但是没接。", "I called her, but I didn't answer.",
     {"refined": "I called her, but she didn't pick up.", "changed": True,
      "reason": "subject dropped in source; translation wrongly assumed 'I'"}),
    ("zh", "en", "这周作业太多了，我直接摆烂了。",
     "There's too much homework this week, I just let it rot.",
     {"refined": "I had way too much homework this week, so I just gave up on it.",
      "changed": True, "reason": "摆烂 is slang for giving up; translated literally"}),
    ("en", "zh", "That party was lit.", "那个派对被点燃了。",
     {"refined": "那个派对太嗨了。", "changed": True,
      "reason": "'lit' is slang for exciting; translated as literally set on fire"}),
    ("en", "zh", "Thank you so much for your help.", "非常感谢你的帮助。",
     {"refined": "非常感谢你的帮助。", "changed": False,
      "reason": "accurate and natural"}),
]


def format_input(src, tgt, source, raw):
    return (f"Source ({LANG_NAMES[src]}): {source}\n"
            f"Machine translation ({LANG_NAMES[tgt]}): {raw}")


def build_messages(src, tgt, source, raw):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_src, ex_tgt, ex_source, ex_raw, ex_out in FEW_SHOT:
        messages.append({"role": "user",
                         "content": format_input(ex_src, ex_tgt, ex_source, ex_raw)})
        messages.append({"role": "assistant",
                         "content": json.dumps(ex_out, ensure_ascii=False)})
    messages.append({"role": "user", "content": format_input(src, tgt, source, raw)})
    return messages


def fix_chinese_punctuation(text):
    """MarianMT outputs half-width punctuation in Chinese. Plain replace, no LLM needed."""
    for half, full in {",": "，", "!": "！", "?": "？", ":": "：", ";": "；"}.items():
        text = text.replace(half, full)
    return text


def refine(src, source, raw):
    tgt = "en" if src == "zh" else "zh"
    if tgt == "zh":
        raw = fix_chinese_punctuation(raw)
    response = ollama.chat(model=MODEL,
                           messages=build_messages(src, tgt, source, raw),
                           format="json",
                           think=False,  # with thinking on, qwen3.5 returns empty content
                           options={"temperature": 0})  # 0 so reruns give the same output
    try:
        content = response["message"]["content"].strip()
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```")
        out = json.loads(content)
        return out["refined"], out["changed"], out["reason"]
    except (json.JSONDecodeError, KeyError):       
        return raw, False, "PARSE_ERROR"


df = pd.read_csv("data/baseline_results.csv", encoding="utf-8-sig", keep_default_na=False)

results = []
for _, row in df.iterrows():
    refined, changed, reason = refine(row["source_lang"], row["source"],
                                      row["raw_translation"])
    results.append((refined, changed, reason))
    print(f"[{row['id']}] {row['raw_translation']}\n"
          f"    -> {refined}  ({'changed' if changed else 'kept'}: {reason})")

df["refined"], df["changed"], df["reason"] = zip(*results)
df["refined_rating"] = ""   
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"\nSaved {len(df)} rows to {OUT_PATH}")
