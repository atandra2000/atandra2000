# Atandra Bharati

Deep Learning Research Engineer. I build frontier AI architectures from scratch in raw PyTorch — no Trainer, no Lightning, every layer by hand — and I write code the way a reviewer would want to read it: documented, tested, and reproducible.

I'm open to Deep Learning, LLM, and GenAI engineering roles (remote or on-site, worldwide).

- Portfolio: https://atandra2000.github.io/mycv
- LinkedIn: https://www.linkedin.com/in/atandrabharati
- Email: atandra.bharati@gmail.com

## About

Self-taught. B.Tech Civil Engineering, 2024. I moved into deep learning by implementing it rather than watching it: starting from GANs and VAEs, up through LLM pretraining, latent diffusion, and multi-agent systems. Civil engineering taught me constraint thinking — models must hold under memory and compute budgets the same way structures hold under load — and that framing shows up in every project below.

**Portfolio (2026):** 29 public repos. All model repos use pure PyTorch (no HF Trainer / Lightning), CPU-friendly pytest gates (`-m "not gpu and not slow"`), and paper-faithful reproduction with regression tests for every bug found. All 9 active LLM repos now have GitHub Actions CI (CPU-only runners on PR + push).

## Selected work

### LLMs (8-model from-scratch portfolio — all single-GPU / A100 80GB targets)

| Project | What it is | Details | Updated |
|---|---|---|---|
| **[HyMo](https://github.com/atandra2000/HyMo)** | Hybrid LLM: gated Delta Net (linear attention) × MLA, asymmetric MoE, MTP. Pretrained from scratch on 30B tokens with FSDP-2 and a custom Triton GDN kernel. | 434M active / 1.13B stored params; 1★ | 2026-09-18 |
| **[DeepSeek-v3-Lite](https://github.com/atandra2000/DeepSeek-v3-Lite)** | From-scratch reimplementation of DeepSeek-V3: MLA, aux-loss-free MoE, MTP, speculative decoding. Includes a 1,950-line technical reference on MLA. | ~412M params, Chinchilla-optimal 8.4B-token run; **3★** | 2026-09-18 |
| **[LLaMA-3-Lite](https://github.com/atandra2000/LLaMA-3-Lite)** | LLaMA 3-style transformer for single-GPU pretraining. GQA, RoPE, Flash Attention, chunked cross-entropy, disk-backed token cache. | 78% peak-memory reduction (92 GB → 20 GB) on a single A100; 1★ | 2026-09-18 |
| **[GPT-OSS-Lite](https://github.com/atandra2000/GPT-OSS-Lite)** | Reproduction of GPT-OSS: sliding/full attention alternation, learned attention sinks, YaRN 128K, top-2-of-8 MoE. | 502M total / 247M active; 1★ | 2026-09-18 |
| **[Mamba-3-Lite](https://github.com/atandra2000/Mamba-3-Lite)** | Mamba-3 reproduction: complex64 SSD, MIMO inter-head mixing, zero causal convolution. Pure PyTorch, no custom CUDA. | ~434M params, 8B-token run | 2026-09-18 |
| **[HiLS-Attention-Lite](https://github.com/atandra2000/HiLS-Attention-Lite)** | Learned sparse chunk attention (~341M GQA): landmark-guided chunk routing, causal top-k selection, score-fused attention, end-to-end trainable by LM loss. | 68 CPU tests passing; new | 2026-09-18 |
| **[DiffusionGemma-Lite](https://github.com/atandra2000/DiffusionGemma-Lite)** | Block-autoregressive discrete diffusion LM: uniform-state diffusion over vocabulary, 256-token canvas, block-causal attention, entropy-bounded adaptive stopping. | 343,516,160 params exactly; new | 2026-09-18 |
| **[deepseek-v41-flash-lite](https://github.com/atandra2000/deepseek-v41-flash-lite)** | DeepSeek-V4.1-Flash replica with CED + CSA2 + hierarchical sparse indexer + SWA + mHC + Engram + MoE + DSpark + ViT. | ~1.16B total / ~205M active; 4×A100 target ≤$500; new | 2026-09-18 |
| **[TranslationLM](https://github.com/atandra2000/TranslationLM)** | Encoder-decoder transformer for English→Italian translation with word-level tokenizers on opus_books. | Full seq2seq from scratch; 6.17→2.28 loss | 2026-08-20 |

### Vision & generative

| Project | What it is | Details | Updated |
|---|---|---|---|
| **[StableDiffusion](https://github.com/atandra2000/StableDiffusion)** | Stable Diffusion 1.x-class latent diffusion trained from scratch: full UNet, DDPM/DDIM, DDP + BF16 on 2× RTX 5090. | 860M UNet, 1.3M+ images across 7 phases | 2026-06-27 |
| **[ActionRecognition](https://github.com/atandra2000/ActionRecognition)** | Skeleton-based action recognition: from-scratch HRNet-like pose estimation + two-stream ST-GCN. Served with ONNX, TensorRT, FastAPI. | ~30 FPS on RTX 3090 | 2026-06-27 |
| **[detect-objects](https://github.com/atandra2000/detect-objects)** | Real-time detector: ResNet-50+FPN with a deformable set-prediction decoder (RT-DETR/DINO-style), no anchors, no NMS. | COCO 2017, ONNX export | 2026-07-18 |
| **[upscale-sr](https://github.com/atandra2000/upscale-sr)** | 4× real-world super-resolution: StableSR-style latent-diffusion UNet + mamba-ssm refiner. | ~1.5s per image on RTX 5090; 1★ | 2026-07-27 |
| **[VisionLanguageModel](https://github.com/atandra2000/VisionLanguageModel)** | PaliGemma-inspired VLM: SigLIP encoder, GQA decoder with RoPE, linear projector. Zero pretrained weights. | COCO 2014 | 2026-08-20 |
| **[FaceAgingCycleGAN](https://github.com/atandra2000/FaceAgingCycleGAN)** | Bidirectional face aging with AdaIN-conditioned CycleGAN, multi-scale discriminator, VGG perceptual loss. | IMDB-Wiki | 2026-06-27 |
| **[DCGAN-Face-Generation](https://github.com/atandra2000/DCGAN-Face-Generation)** | DCGAN on CelebA with experiment tracking, from scratch. | 64×64 faces | 2026-06-27 |
| **[FaceGenerationVAE](https://github.com/atandra2000/FaceGenerationVAE)** | β-VAE for face generation: bilinear upsampling, KL annealing, 512-dim latent. | CelebA, Comet tracking | 2026-08-20 |

### Agentic systems

| Project | What it is | Details | Updated |
|---|---|---|---|
| **[AutonomousResearcher](https://github.com/atandra2000/AutonomousResearcher)** | Multi-agent platform that reads papers, plans experiments, writes and runs training code, and iterates. Provider-agnostic LLM routing, 15 phases, 23 agents, 61 tools, 878 passing tests. | 1★ | 2026-08-29 |
| **[news-agent](https://github.com/atandra2000/news-agent)** | Autonomous research-intelligence agent: daily multi-source sweep, LLM synthesis with self-critique, provenance-tracked reports, 291 offline tests. | — | 2026-07-20 |

### Also

| Project | What it is | Updated |
|---|---|---|
| **[CyberLM](https://github.com/atandra2000/CyberLM)** | Sovereign cybersecurity agentic MoE: 4.55B total / 1.34B active, air-gapped, RAG + GRPO + SFT trajectory engine. Not yet public — local repo only. | 2026-09-19 |
| **[LearnAgenticAI](https://github.com/atandra2000/LearnAgenticAI)** | Ten progressive agentic-AI projects with LangChain, LangGraph, LangSmith, MCP, and a Next.js chat UI. | 2026-08-31 |
| **[FusionLLM](https://github.com/atandra2000/FusionLLM)** | Hybrid LLM pretraining framework: MLA + gated Delta Net + DeepSeek MoE + MTP. 415M active params. **Archived** (superseded by HyMo). | 2026-07-17 |
| **[GPT-From-Scratch](https://github.com/atandra2000/GPT-From-Scratch)** | Character-level GPT on Tiny Shakespeare. The project that started the portfolio (frozen educational). | 2026-08-20 |
| **[qwen3.8-guide](https://github.com/atandra2000/qwen3.8-guide)** | Interactive first-principles explainer of the Qwen 3.8-27B architecture (36 interactive widgets). | 2026-08-24 |
| **[flash-attention-2-guide](https://github.com/atandra2000/flash-attention-2-guide)** | Interactive FlashAttention-2 field guide: online softmax, SRAM tiling, backward pass (dependency-free HTML). | 2026-08-21 |

## Writing

- [Multi-Head Latent Attention & Mixed Precision — 1,950-line technical deep-dive](https://github.com/atandra2000/DeepSeek-v3-Lite/blob/main/docs/concepts/attention-and-precision.md) — KV-cache compression math, absorption trick, decoupled RoPE.
- [Qwen 3.8-27B architecture guide](https://github.com/atandra2000/qwen3.8-guide) — derived from first principles.
- [FlashAttention-2 guide](https://github.com/atandra2000/flash-attention-2-guide) — online softmax, SRAM tiling, backward pass.

## CI / automation

Every active model repo runs a CPU-only GitHub Actions workflow (`ci.yml`) that fires on PR + push to `main`: installs Python 3.10–3.12, installs `torch` (CPU index) + `pytest`, and runs `pytest -m "not gpu and not slow"`. This is the project convention — GPU tests are marker-gated (`gpu` / `slow` / `heavy`) and skip locally on macOS / CPU.

Repos covered: DeepSeek-v3-Lite, LLaMA-3-Lite, GPT-OSS-Lite, HyMo, Mamba-3-Lite, TranslationLM, deepseek-v41-flash-lite, HiLS-Attention-Lite, DiffusionGemma-Lite, GPT2.

## Skills

**Languages:** Python, SQL, JavaScript
**ML:** PyTorch 2.x (BF16/FP16/FP8, SDPA, Flash-Attention 2, torch.compile), HuggingFace, diffusers, CUDA, DDP, FSDP-2
**Agentic:** LangChain, LangGraph, LangSmith, MCP, pydantic v2
**Tracking & serving:** W&B, Comet, ONNX, TensorRT, FastAPI
**Infrastructure:** Linux, Docker, Git

## Engineering practices

- From-scratch PyTorch everywhere; no Trainer, Lightning, or accelerate.
- Every project tracked on W&B or Comet with full RNG-state reproducibility and atomic checkpoint writes.
- Faithful paper reproduction: DeepSeek-V3, LLaMA-3, GPT-OSS, Mamba-3, PaliGemma, HiLS, DiffusionGemma — implemented to the paper, with regression tests for every bug found.
- All model repos ship with design specs (`DESIGN-*.md`), execution plans (`EXECUTION-PLAN-*.md`), source manifests (`source-manifest.md`), and interactive HTML visual guides.

## Background

B.Tech Civil Engineering, Heritage Institute of Technology, Kolkata, 2024. The transition from civil to ML is a strength, not a gap: engineering discipline under constraint — memory budgets, distributed training, reproducibility — transfers directly to deep learning systems.
