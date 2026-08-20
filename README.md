# Minimal SynthID-Text Watermark

A minimal, educational implementation of the published SynthID-Text watermarking baseline for Hugging Face causal language models.

This project is inspired by Anthropic's article, [How Claude’s text watermark works](https://www.anthropic.com/news/claude-text-watermark), which explains that future Claude models will use a version of Google DeepMind's SynthID-Text approach. Anthropic has not published Claude's exact production implementation, key, or detector; this repository instead follows the public [SynthID-Text paper](https://www.nature.com/articles/s41586-024-08025-4).

The goal is to show the complete watermarking flow with as little code and as few dependencies as possible:

Generation and ordinary detection use `torch` and `transformers`. The paired
calibration generator additionally uses `datasets`; `accelerate` is
needed when using Transformers automatic device placement.

1. Generate and save a secret watermark key.
2. Embed a statistical watermark during token sampling.
3. Detect the watermark using the same key and tokenizer.

## Working

### Generation

At each step, the watermark uses:

- a secret 32-byte key;
- the previous four token IDs as context;
- three tournament layers by default, which means eight sampled candidates.

1. The model produces next-token logits for the current context.
2. Temperature and top-p filtering produce the normal sampling distribution.
3. Eight candidate tokens are sampled from that distribution.
4. For each candidate and tournament layer, a keyed hash of the key, four-token context, layer, and candidate token produces a deterministic bit.
5. Candidates compete in a tournament that statistically favors higher-scoring tokens.
6. The winning token is appended and becomes part of the context for the next step.

### Normal vs. watermarked sampling

Normal stochastic sampling:

```text
current context
    ↓
model logits
    ↓
temperature/top-p filtering
    ↓
sample one token from the probabilities
```

Watermarked generation:

```text
current context
    ↓
model logits
    ↓
temperature/top-p filtering
    ↓
sample several candidate tokens
    ↓
keyed tournament
    ↓
choose one candidate
```

The main difference is:

```text
normal sampling:
    probability distribution → sample one token

watermarked sampling:
    probability distribution → sample several tokens
    → use the key to select a statistically favorable candidate
```

### Detection

1. Tokenize the text with the same tokenizer used for generation.
2. Recompute the keyed watermark bits for each token and its preceding
   context.
3. For calibrated scoring, keep the first configured number of
   response-text tokens and reject shorter responses.
4. Skip later occurrences of a four-token context that has already been
   scored in the response.
5. Aggregate the remaining bits into a raw watermark score and report the
   number of unique scored contexts.

Detection only needs the text, tokenizer, key, and watermark configuration - not the language model. A correct key should produce an elevated score for watermarked text, while ordinary text or a wrong key should remain near the random baseline.

Raw scores are evidence, not a classification decision. Classification requires
a saved calibration JSON artifact whose metadata matches the detector
configuration; a missing or incompatible artifact is an error rather than an
uncalibrated fallback.

This repository is educational and non-production. It does not claim to
reproduce any private production watermark implementation or provide
production-grade calibration guarantees.

## Command-line tools

Generate the default 256-bit key at `keys/watermark.key`:

```bash
uv run python -m scripts.generate_key
```

## Calibrated test-set evaluation

Raw scores are not calibrated thresholds. Generate separate development and
test JSONL files, fit the threshold using only ordinary development responses,
then evaluate the frozen artifact on held-out ordinary and watermarked test
responses:

```bash
uv run python -m scripts.generate_data \
    --model Qwen/Qwen3-14B-FP8 \
    --development-count 1200 \
    --test-count 400 \
    --max-new-tokens 1024 \
    --temperature 0.7 \
    --top-p 0.8 \
    --device-map auto \
    --batch-size 4 \
    --output-dir outputs/calibration

uv run python -m scripts.fit_threshold \
    --development outputs/calibration/development.jsonl \
    --output outputs/calibration/calibration.json \
    --required-tokens 200 \
    --target-fpr 0.01

uv run python -m scripts.evaluate \
    --test outputs/calibration/test.jsonl \
    --calibration outputs/calibration/calibration.json \
    --predictions-output outputs/calibration/test-predictions.jsonl \
    --metrics-output outputs/calibration/test-metrics.json
```

`generate_data` writes two files. Development prompts produce one ordinary
row each. Test prompts produce one ordinary and one watermarked row each:

`--batch-size` controls how many prompts are generated in one model batch;
use `1` to disable batching when memory is limited.

```text
outputs/calibration/development.jsonl  # development-count rows
outputs/calibration/test.jsonl         # 2 × test-count rows
```

`fit_threshold` uses only rows with `kind: "unwatermarked"` from the
development file to choose the upper-tail threshold. It tokenizes each row's
`text` using the declared tokenizer, then scores exactly the first 200
tokenized response tokens; shorter responses are insufficient.

`evaluate` loads the frozen threshold and applies it to every labeled test row.
It writes one prediction JSON object per input row, preserving the original
row fields and adding the row number, score, predicted kind, correctness, and
token-count details. It also reports aggregate test FPR and TPR.

Calibration rows store response text rather than generated token IDs. Fitting
and evaluation tokenize that text with the calibration tokenizer, so both
stages use the same text-to-token pipeline. The calibration artifact stores
the fitted threshold, key fingerprint, model, tokenizer, tournament layers,
and required response-token prefix length.

