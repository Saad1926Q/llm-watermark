# Minimal SynthID-Text Watermark

A minimal, educational implementation of the published SynthID-Text watermarking baseline for Hugging Face causal language models.

This project is inspired by Anthropic's article, [How Claude’s text watermark works](https://www.anthropic.com/news/claude-text-watermark), which explains that future Claude models will use a version of Google DeepMind's SynthID-Text approach. Anthropic has not published Claude's exact production implementation, key, or detector; this repository instead follows the public [SynthID-Text paper](https://www.nature.com/articles/s41586-024-08025-4).

The goal is to show the complete watermarking flow with as little code and as few dependencies as possible:

1. Generate and save a secret watermark key.
2. Embed a statistical watermark during token sampling.
3. Detect the watermark using the same key and tokenizer.
