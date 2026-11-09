---
inclusion: fileMatch
fileMatchPattern: "config.py"
---

# config.py Rules

`config.py` is the single source for paths and parameters. Stage scripts import it as `import config`.

- Keep the existing section headers (Global, Stage 1 to 3). Add new settings under the matching header.
- Keep `NORMALIZATION_COLUMNS` in sync with `preprocess/normalization_stats.json`. Update the comment above the list whenever the list changes.
- `VAL_RATIO` comment: it is set to `0.0` for inference-only conversion. Do not change it without asking.
- Do not add secrets or credentials to this file.
- Changing any setting that affects features or shards makes existing outputs stale. Mention which outputs need regeneration when you change one.
