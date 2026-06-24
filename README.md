# Atandra Bharati

**Deep learning research engineer** rebuilding frontier AI architectures from scratch in PyTorch — LLMs, latent diffusion, multimodal, and video understanding. Self-taught; no high-level wrappers, single-GPU heroics, paper-faithful reproductions.

## Anchor results

- **78% peak memory reduction (92 GB → 20 GB)** for LLM pretraining via gradient checkpointing, chunked cross-entropy, and disk-backed token caching — enabling 2× batch-size headroom on a single A100 80GB.
- **Best training loss 0.0947 at epoch 16** training Stable Diffusion 1.x from scratch (860M UNet) over a 7-phase curriculum on 2× RTX 5090.
- **11 end-to-end projects** across LLMs, generative vision, multimodal AI, and video — each engineered for single-GPU feasibility (A100, RTX 5090, RTX 6000 Ada, RTX 3090, P100, T4) with BF16, Flash Attention 2, gradient checkpointing, and `torch.compile`.

## Flagship architectures

These are the deepest architectural reproductions — full frontier stacks rebuilt from first principles on a single GPU.

| Project | What was rebuilt | Key innovation | Link |
|---------|------------------|----------------|------|
| **DeepSeek-v3-Lite** (422M) | MLA + aux-loss-free MoE + MTP, end-to-end | True MLA absorption at inference, μP-scaled LR, 8.4B-token Chinchilla-optimal recipe | [repo](https://github.com/atandra2000/DeepSeek-v3-Lite) |
| **LLaMA-3-Lite** (515M) | GQA, RoPE, fused SwiGLU, RMSNorm, Flash-Attn 2 | 78% peak-memory reduction; chunked CE (logits 50 GB → 0.3 GB) | [repo](https://github.com/atandra2000/LLaMA-3-Lite) |
| **FusionLLM** (415.6M active / 868.6M stored) | MLA + Gated Delta Net + MoE + MTP in one 24-layer hybrid | Dual optimizer (NorMuon + CautiousAdamW), WSD + μP, logit softcap | [repo](https://github.com/atandra2000/FusionLLM) |

*Architecture and training pipelines complete; large-scale pre-training not yet started.*

## Verified builds

Projects that have completed training runs with tracked metrics.

| Project | Outcome | Link |
|---------|---------|------|
| **Stable Diffusion 1.x** (860M UNet) | Best loss **0.0947** at epoch 16; 42-epoch run released on 2× RTX 5090 across LAION, DiffusionDB, JourneyDB, VGGFace2, COCO | [repo](https://github.com/atandra2000/StableDiffusion) |
| **VisionLangModel** (PaliGemma-style) | Trained end-to-end on COCO 2014 captions, zero pre-trained weights, single P100 | [repo](https://github.com/atandra2000/VisionLanguageModel) |
| **FaceAgingCycleGAN** (AdaIN-conditioned) | 31 epochs on IMDB-WIKI; per-layer age conditioning, 3-scale PatchGAN, RTX 6000 Ada | [repo](https://github.com/atandra2000/FaceAgingCycleGAN) |
| **FaceGenerationVAE** (β-VAE) | 50 epochs on CelebA; recon MSE 0.0152, KL annealing 0→1, bilinear-upsample decoder | [repo](https://github.com/atandra2000/FaceGenerationVAE) |
| **DCGAN-Face-Generation** | 50 epochs on 202k CelebA; D loss → ln 2 ≈ 0.693 equilibrium, 2× T4 | [repo](https://github.com/atandra2000/DCGAN-Face-Generation) |
| **TranslationLM** (EN→IT seq2seq) | 20 epochs on OPUS Books; cross-attention visualizations, custom SentencePiece BPE | [repo](https://github.com/atandra2000/TranslationLM) |
| **GPT-From-Scratch** | 200-line educational GPT-2, trained on Tiny Shakespeare | [repo](https://github.com/atandra2000/GPT-From-Scratch) |


## Autonomous Research Platform

| Project | Architecture | Key metrics | Link |
|---------|-------------|-------------|------|
| **Autonomous ML Research Engineer** | 15-phase multi-agent platform: literature discovery, paper analysis, repo analysis, experiment planning, code implementation, training execution, evaluation, autonomous looping, self-repair, end-to-end research workflows | 23 agents, 61 tools, 186 models, 878 tests, provider-agnostic LLM layer with per-agent routing (Qwen3-coder, GLM-5.2, MiniMax-M3) | [repo](https://github.com/atandra2000/AutonomousResearcher) |

## In-progress pipelines

| Project | Status | Link |
|---------|--------|------|
| **ActionRecognition** (ST-GCN) | Pose estimation + ST-GCN pipeline ready; NTU RGB+D 120 benchmark not started | [repo](https://github.com/atandra2000/ActionRecognition) |

## Writing

- **"Multi-Head Latent Attention — A Technical Deep-Dive"** — 643-line reference covering KV-cache math, low-rank compression algebra, the absorption-trick derivation, decoupled RoPE mechanics, and SDPA vs manual attention path trade-offs in DeepSeek-V2/V3. ([`MLA.md`](https://github.com/atandra2000/DeepSeek-v3-Lite/blob/main/MLA.md))

## Links

- Portfolio site: [atandra2000.github.io/mycv](https://atandra2000.github.io/mycv)
- LinkedIn: [linkedin.com/in/atandrabharati](https://www.linkedin.com/in/atandrabharati)
- Weights & Biases: [wandb.ai/atandrabharati-self](https://wandb.ai/atandrabharati-self)
- Kaggle: [kaggle.com/atandrabharati](https://kaggle.com/atandrabharati)
- Comet: [comet.com/atandrabharati](https://comet.com/atandrabharati)

