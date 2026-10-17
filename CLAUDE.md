# Intelligent Translator

A multilingual speech translator with an AI "refinement" layer that fixes mistranslations
so the output sounds natural (slang, idioms, dropped subjects, register). Started with
English ↔ Mandarin; expanding to the most widely spoken languages.

**The full plan is in `ROADMAP.md`. Read it before starting any new phase.**

## Pipeline

mic → Whisper (detects language) → translation → LLM refinement (Qwen via Ollama) → TTS

Only the text stages exist so far. Translation is MarianMT now; NLLB-200 replaces it in Phase 2.

## Current status

- Phase 1 in progress: test set and baseline done; NEXT is running `refine.py`,
  rating the output, and comparing before/after by category.

## Multilingual principles

- All language info lives in one config (`languages.py`, Phase 2). No hardcoded language codes elsewhere.
- Languages are not hand-tested one by one. Automatic FLORES+ evaluation (chrF + COMET,
  raw vs refined) decides each language's `refine` and `quality` settings.
- Refinement can make weak-language output fluent but wrong. Only enable it where scores show it helps.

## Files

- `test_set.csv`: columns `id, source_lang, category, term, source, intended_meaning, meaning_verified, verify_source`
  - `source` = the input sentence; `term` = the slang/idiom being tested; `intended_meaning` = what a good translation of the whole sentence should say (in English)
- `baseline_translate.py`: runs MarianMT (`Helsinki-NLP/opus-mt-zh-en` and `opus-mt-en-zh`; en→zh needs the `>>cmn_Hans<<` prefix)
- `refine.py`: sends source + raw translation to Qwen with a system prompt and few-shot examples; outputs JSON `{refined, changed, reason}`; falls back to raw translation on parse errors
- `requirements.txt`

## Key decisions (don't undo without asking)

- The refinement model **post-edits**, it doesn't rewrite: minimal changes, leave correct translations alone, never add information.
- Few-shot examples in `refine.py` must **never** use sentences from the test set (data leakage).
- Rating scale is exactly `good` / `awkward` / `wrong`. Rule: "if a listener heard only this translation, would they understand what was meant?" If the key fact is wrong (e.g. wrong person in a dropped-subject sentence), it's `wrong`.
- Deterministic fixes (e.g. Chinese full-width punctuation) are done in plain code, not by the LLM.
- `temperature: 0` for repeatable results.
- Local open-source model first (free, private); an API model may be added later as a comparison.

## Evaluation constraint

I speak and understand Mandarin only at a basic level, can't read or write it, and don't know Chinese slang.
So I can't judge Chinese text quality. Ground truth comes from:
- `intended_meaning` (English), verified against sources online
- human-translated datasets (FLORES-200, Tatoeba) with chrF/COMET
- a fluent speaker spot-checking a sample
- an LLM judge that is a *different* model from the refiner, checked against human ratings
Don't present your own explanations of Chinese as verified fact.

## Environment

- Windows, PowerShell, VS Code. Project lives at `C:\dev\IntelligentTranslatorProject` (moved out of OneDrive).
- Virtual env in `.venv` (activate with `.venv\Scripts\activate`). Had a "filename too long" error installing torch with Microsoft Store Python; fix was enabling Windows long paths and keeping the project at a short path.
- Ollama model: `qwen3.5:9b`. If it outputs reasoning that breaks JSON, pass `think=False` to `ollama.chat`.
- ~32 GB RAM.

## How to work with me

I'm a CS student learning ML, and I want to understand everything I build.
- Explain the concept and your reasoning before showing code or making edits.
- Propose changes and wait for my OK before editing files.
- Prefer small steps; let me try things myself when practical.
- Be precise. If something you said earlier was wrong, say so.
- Use git: small commits with clear messages.