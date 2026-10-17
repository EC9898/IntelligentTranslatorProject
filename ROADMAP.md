# Roadmap: multilingual speech translator

Goal: speech and text translation across the most widely spoken languages, with an LLM
refinement layer that makes output sound natural. Quality is **measured per language
automatically**, and the app's behavior per language follows from those measurements.

Work through the phases in order. Finish and commit each step before starting the next.

---

## Phase 1: Finish English ↔ Mandarin (current)

1. Run `refine.py` on `baseline_results.csv` with `qwen3.5:9b`.
2. Rate the refined zh→en rows with the same good/awkward/wrong rule.
3. Write `compare.py`: counts of good/awkward/wrong per category, baseline vs refined,
   plus how often refinement changed a correct translation (over-edits).
4. Record the results in the README.

Done when: a before/after table by category exists and is committed.

## Phase 2: Make the code language-agnostic

1. Add `languages.py` with one config dict, one entry per language:
   ```python
   "zh": {
       "name": "Mandarin Chinese",
       "whisper": "zh",          # Whisper's language code
       "nllb": "zho_Hans",       # NLLB / FLORES code
       "tts_voice": "zh-CN-XiaoxiaoNeural",
       "speech_in": True,        # filled in after testing
       "speech_out": True,
       "refine": True,           # set by the Phase 3 evaluation
       "quality": None,          # "good" / "ok" / "low", set by Phase 3
   }
   ```
   Every script reads language info from here. No language codes hardcoded elsewhere.
2. Remove the two-language assumptions:
   - `tgt = "en" if src == "zh" else "zh"` → source and target are both parameters
   - language-specific cleanup (e.g. Chinese full-width punctuation) runs only when that
     language is the target, via a per-language hook
3. Replace MarianMT with **NLLB-200** (`facebook/nllb-200-distilled-600M` to start).
   One model, any direction. License is non-commercial (fine for this project).
4. **Regression check**: re-run the Mandarin test set through NLLB and confirm it isn't
   worse than the MarianMT baseline. Keep MarianMT results for comparison.

Done when: adding a language means adding one config entry and nothing else.

## Phase 3: Automatic evaluation for every language

No hand-testing per language. The evaluation decides each language's settings.

1. Dataset: **FLORES+** (maintained version of FLORES-200) from Hugging Face.
   Same sentences professionally translated into 200 languages. May require accepting
   terms on the dataset page.
2. Directions: evaluate **X → English** and **English → X** only, not every pair
   (15 languages would be 210 pairs). English acts as the hub.
3. Sample ~100 sentences per direction (use the same IDs for every language).
4. For each language and direction, translate with NLLB, then refine with Qwen.
5. Score both raw and refined output against the human reference:
   - **chrF** (`sacrebleu`): fast, run everywhere
   - **COMET** (`unbabel-comet`): better at meaning, slow on CPU; run in Colab with a GPU
     if needed
6. Save everything to `results/flores_scores.csv`
   (language, direction, system, chrF, COMET, n_sentences).
7. Set per-language config from the scores:
   - `refine`: True only if refinement improves COMET for that language
   - `quality`: tier based on the scores
   Choose the thresholds **after** looking at the score distribution, and document them.
8. Watch for refinement making scores *worse*: that's the "fluent but wrong" failure.

Done when: every language in the config has `refine` and `quality` set from evidence.

## Phase 4: Speech in and out

1. Whisper (`faster-whisper`, `small` or `medium`) for speech-to-text with language
   detection. Set `speech_in` per language from a quick check.
2. TTS with `edge-tts`. Verify each voice exists with `edge-tts --list-voices`.
   Set `speech_out` per language.
3. Full pipeline function: audio in → detect language → translate to chosen target →
   refine (if enabled) → speak.

## Phase 5: Interface

1. Gradio app: microphone input, target-language dropdown, text + audio output.
2. Show a warning for languages with `quality == "low"`.
3. Offer text-only translation for languages without speech support.

## Later / optional

- Hand-built slang test sets for 2–3 more languages (sourced meanings, ideally a fluent
  speaker checking a sample)
- API-model comparison for refinement (local vs hosted)
- LLM judge (a different model from the refiner), checked against human ratings
- Conversation context (previous turns) in the refinement prompt

---

## Starting language list

Among the most widely spoken languages, all supported by Whisper and NLLB.
Verify every code and voice before using it.

| Code | Language | NLLB code |
|---|---|---|
| en | English | eng_Latn |
| zh | Mandarin Chinese (Simplified) | zho_Hans |
| hi | Hindi | hin_Deva |
| es | Spanish | spa_Latn |
| fr | French | fra_Latn |
| ar | Arabic (Modern Standard) | arb_Arab |
| bn | Bengali | ben_Beng |
| pt | Portuguese | por_Latn |
| ru | Russian | rus_Cyrl |
| ur | Urdu | urd_Arab |
| id | Indonesian | ind_Latn |
| de | German | deu_Latn |
| ja | Japanese | jpn_Jpan |
| ko | Korean | kor_Hang |
| vi | Vietnamese | vie_Latn |
| tr | Turkish | tur_Latn |
| sw | Swahili | swh_Latn |
| ta | Tamil | tam_Taml |

Notes:
- Arabic: FLORES/NLLB use Modern Standard Arabic, but people *speak* regional dialects
  that differ a lot. Expect speech results to be weaker than text scores suggest.
- Hindi and Urdu are very similar when spoken but use different scripts.
- Run the evaluation on 3–4 languages first to make sure the pipeline works, then all.
