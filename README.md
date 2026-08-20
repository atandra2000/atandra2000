# Atandra Bharati

**Deep Learning Research Engineer** building frontier AI from scratch in PyTorch — efficient language models, generative vision, and agentic research systems. From-scratch first: no pretrained weights, no wrappers, every architecture reimplemented and documented.

## Flagship work

- **[HyMo](https://github.com/atandra2000/HyMo)** — Hybrid LLM fusing Gated Delta Net (linear attention) × MLA (full attention) with Asymmetric MoE. 434M active / 1.13B stored params, pre-trained from scratch on 30B tokens, FSDP-2 + custom Triton GDN kernel.
- **[DeepSeek-v3-Lite](https://github.com/atandra2000/DeepSeek-v3-Lite)** — Faithful DeepSeek-V3 reimplementation: MLA absorption, aux-loss-free MoE, MTP, speculative decoding. ~412M params, Chinchilla-optimal on a single A100.
- **[LLaMA-3-Lite](https://github.com/atandra2000/LLaMA-3-Lite)** — LLaMA-3-style dense transformer with an 8-technique memory stack: **78% peak-memory reduction** (92 GB → 20 GB) via chunked CE + disk-backed token cache + BF16.
- **[GPT-OSS-Lite](https://github.com/atandra2000/GPT-OSS-Lite)** — OpenAI GPT-OSS reproduction: sliding/full attention alternation, learned attention sinks, YaRN 128K, top-2-of-8 MoE. 502M total / 247M active.
- **[Mamba-3-Lite](https://github.com/atandra2000/Mamba-3-Lite)** — Mamba-3 with complex64 SSD state spaces (N=64), MIMO head mixing, zero causal convolution. Pure PyTorch, no custom CUDA.
- **[StableDiffusion](https://github.com/atandra2000/StableDiffusion)** — 860M-param latent diffusion trained from scratch on 2× RTX 5090; best loss 0.0947.

## By domain

| Domain | Projects |
|---|---|
| Language models | HyMo · DeepSeek-v3-Lite · LLaMA-3-Lite · GPT-OSS-Lite · Mamba-3-Lite · TranslationLM · GPT-From-Scratch |
| Generative vision | StableDiffusion · FaceAgingCycleGAN · FaceGenerationVAE · DCGAN |
| Vision & multimodal | VisionLanguageModel · detect-objects · upscale-sr · ActionRecognition |
| Agentic systems | AutonomousResearcher (15-phase, 23 agents, 878 tests) · news-agent |
| Education | MA1101 · CS1101 · EE1101 (offline-first interactive textbooks) |

## How I work

- **Raw PyTorch** — no HF Trainer, no Lightning; custom Triton kernels only on measured hot paths.
- **Measured, not assumed** — every headline metric ships with a reproducing script or test.
- **Docs ship with code** — machine-verified `docs/` portals on every flagship project.
- **Production discipline** — atomic checkpoints, full RNG state, W&B tracking.

## Connect

- [Portfolio](https://atandra2000.github.io/mycv)
- [LinkedIn](https://www.linkedin.com/in/atandrabharati)
- [Email](mailto:atandra.bharati@gmail.com)

*Open to ML research engineer roles. Kolkata, India · Remote-friendly.*
