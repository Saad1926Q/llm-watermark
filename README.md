# Minimal SynthID-Text Watermark

A minimal, educational implementation of the published SynthID-Text watermarking baseline for Hugging Face causal language models.

This project is inspired by Anthropic's article, [How Claude’s text watermark works](https://www.anthropic.com/news/claude-text-watermark), which explains that future Claude models will use a version of Google DeepMind's SynthID-Text approach. Anthropic has not published Claude's exact production implementation, key, or detector; this repository instead follows the public [SynthID-Text paper](https://www.nature.com/articles/s41586-024-08025-4).

The goal is to show the complete watermarking flow with as little code and as few dependencies as possible:

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
2. Recompute the keyed watermark bits for each token and its preceding context.
3. Aggregate the bits into a watermark score.
4. Report the score, z-score, and approximate p-value.

Detection only needs the text, tokenizer, key, and watermark configuration - not the language model. A correct key should produce an elevated score for watermarked text, while ordinary text or a wrong key should remain near the random baseline.

This provides statistical evidence that text is consistent with the key; it is not proof of authorship.

## Qwen3-14B FP8

Run the comparison with Qwen's official FP8 checkpoint using automatic model placement:

```bash
uv run compare.py \
    --model Qwen/Qwen3-14B-FP8 \
    --device-map auto \
    --output-file outputs/qwen3-14b-fp8.jsonl \
    --max-new-tokens 1024 \
    --temperature 0.7 \
    --top-p 0.8
```

Use a recent NVIDIA GPU with approximately 24 GB of VRAM. Automatic placement also works
with unquantized checkpoints. The checkpoint determines its numerical format; `--device-map`
only controls where Transformers places the model.
