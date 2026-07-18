<div align="center">

# Atandra Bharati

**Deep Learning Research Engineer** — building frontier AI architectures from scratch in raw PyTorch.

LLMs · Latent Diffusion · Multimodal · Video Understanding · Agentic ML · State-Space Models · Long-Context Attention

`16 from-scratch projects` · `78% memory optimization` · `878-test agentic platform` · `860M-param UNet trained from random init`

</div>

---

## 🎯 Open To

**Deep Learning Research Engineer** · **LLM Engineer** · **GenAI / Diffusion Engineer** · **Agentic ML Engineer**

Remote-friendly · Available worldwide

---

## 🧭 Now

Shipping the **Autonomous ML Research Engineer** platform (15 phases, 23 agents) and developing **HyMo** — the flagship hybrid language model (3:1 Gated Delta Net / Multi-Head Latent Attention, Asymmetric MoE, custom Triton GDN kernel, 750M active / 1.86B stored params). Also just released two new long-context / state-space reproductions: **GPT-OSS-Lite** (sliding/full attention alternation + learned sinks, 2× KV-cache cut at 128K) and **Mamba-3-Lite** (complex-valued SSD + MIMO mixing, zero causal conv), plus two new vision products: **Detect-Objects** (deformable set-prediction detector) and **Upscale-SR** (4× diffusion + SSM super-resolution).

---

## 🛠️ Stack

**Languages & ML core**  
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)

**Architectures**  
Transformers · GQA · MLA · Sliding/Full Attention Alternation · Learned Attention Sinks · YaRN RoPE · SwiGLU · RMSNorm · MoE · Gated Delta Net · MTP · SSD (real & complex64) · MIMO head mixing · Diffusion UNet · VAE · GAN · CycleGAN · AdaIN · ST-GCN · HRNet · SigLIP · Deformable Detection · Super-Resolution

**Optimization & numerics**  
BF16 · FP16 · FP8 · Flash Attention 2 · SDPA · `torch.compile` · `channels_last` · Gradient checkpointing · μP scaling · WSD LR · NorMuon · CautiousAdamW · Chunked cross-entropy · Disk-backed token caching · Fused optimizers · Chinchilla-optimal scaling

**Hardware validated**  
A100 80GB · RTX 5090 (Blackwell) · RTX 6000 Ada · RTX 3090 · P100 · 2× T4

**Tooling**  
HuggingFace · diffusers · tiktoken · W&B · Comet · safetensors · ONNX · TensorRT · FastAPI · Gradio · pydantic v2 · ChromaDB · Ollama Cloud

---

## 🏆 Highlights

- **Efficiency at scale.** Cut peak pretraining memory 78% (92 → 20 GB) for 2× batch headroom on one A100 80GB, and halved KV-cache at 128K context (1.13 vs 2.25 GB) via sliding/full attention alternation — cheaper training and cheaper serving.
- **Novel hybrid architecture.** HyMo pairs a 3:1 Gated Delta Net / Multi-Head Latent Attention stack (750M active / 1.86B stored) with Asymmetric MoE, MTP, and a hand-written Triton GDN kernel trained under FSDP-2.
- **Faithful from-scratch reproductions.** Stable Diffusion 1.x UNet reaching loss 0.0947 from random init, plus complex64 SSD matching baseline loss at 50% smaller state (N=64) with MIMO mixing and zero custom CUDA.
- **Shipped, deployable systems.** ~30 FPS skeleton action recognition via ONNX + TensorRT + FastAPI, and an 878-test autonomous research platform (15 phases · 23 agents · 186 models) that runs the full paper-to-conclusions loop.

---

## 📂 Projects

### LLM (7)

| Project | Scale | Highlight | Hardware | Repo |
|---------|-------|-----------|----------|------|
| **HyMo** ⭐ | 750M active / 1.86B stored | 3:1 GDN/MLA hybrid · Asymmetric MoE · MTP · custom Triton GDN kernel · FSDP-2 | 4× A100 80GB | [→](https://github.com/atandra2000/HyMo) |
| **GPT-OSS-Lite** | 502M / 247M active | Sliding(128)/Full attention alt · learned sink bias · YaRN 128K · top-2-of-8 MoE · **2× KV-cache cut at 128K** · **130 tests** | A100 80GB | [→](https://github.com/atandra2000/GPT-OSS-Lite) |
| **Mamba-3-Lite** | ~380M | Complex64 SSD (N=64) · MIMO head mixing · zero causal conv · pure PyTorch | A100 80GB | [→](https://github.com/atandra2000/Mamba-3-Lite) |
| **DeepSeek-v3-Lite** | ~422M | MLA + AuxLossFreeGate MoE + MTP, end-to-end with absorption-trick inference · **643-line MLA ref** | A100 80GB | [→](https://github.com/atandra2000/DeepSeek-v3-Lite) |
| **LLaMA-3-Lite** | ~515M | GQA · RoPE θ=500K · SwiGLU · RMSNorm · FA2 · chunked CE · **78% memory cut** | A100 80GB | [→](https://github.com/atandra2000/LLaMA-3-Lite) |
| **TranslationLM** (EN→IT) | ~44M | Encoder–decoder Transformer · loss 6.17 → 2.28 · BLEU/CER/WER | P100 | [→](https://github.com/atandra2000/TranslationLM) |
| **GPT-2** | ~124M | Educational nanoGPT-style decoder · tiktoken BPE · HF weight loading | MPS / CUDA / CPU | [→](https://github.com/atandra2000/GPT2) |

### Vision (8)

| Project | Scale | Highlight | Hardware | Repo |
|---------|-------|-----------|----------|------|
| **Stable Diffusion 1.x** | 860M UNet | Custom UNet trained from random init · 7 phases · 1.3M+ images · **best loss 0.0947** · epoch-42 checkpoint | 2× RTX 5090 | [→](https://github.com/atandra2000/StableDiffusion) |
| **Detect-Objects** | ~50M | RT-DETR/DINO-style deformable set-prediction detector · no anchors/NMS · COCO 2017 · Gradio + ONNX | 2× RTX 5090 | [→](https://github.com/atandra2000/detect-objects) |
| **Upscale-SR** | ~75M | 4× real-world photo SR · latent-diffusion UNet + SSM refiner · Real-ESRGAN degradation | 2× RTX 5090 | [→](https://github.com/atandra2000/upscale-sr) |
| **ActionRecognition** | 120 cls | HRNet pose + Two-Stream CTR-GCN · **~30 FPS** · ONNX + TensorRT | RTX 3090 | [→](https://github.com/atandra2000/ActionRecognition) |
| **FaceAgingCycleGAN** | 256² | Per-layer AdaIN conditioning · 3-scale PatchGAN · LSGAN + R1 GP | RTX 6000 Ada | [→](https://github.com/atandra2000/FaceAgingCycleGAN) |
| **FaceGenerationVAE** | β-VAE | 50 epochs · recon MSE 0.0152 · linear KL annealing · bilinear-upsample decoder | P100 | [→](https://github.com/atandra2000/FaceGenerationVAE) |
| **DCGAN-Face-Generation** | 6.4M | 50 epochs · 202K CelebA · D loss → ln 2 ≈ 0.693 equilibrium | 2× T4 | [→](https://github.com/atandra2000/DCGAN-Face-Generation) |
| **VisionLangModel** | PaliGemma-style | SigLIP ViT + Gemma decoder + linear projector · zero pretrained weights | P100 | [→](https://github.com/atandra2000/VisionLangModel) |

### Agentic (2)

| Project | Highlight | Hardware | Repo |
|---------|-----------|----------|------|
| **Autonomous ML Research Engineer** | 15-phase multi-agent platform · paper → plan → patch → train → evaluate → report · 23 agents · 61 tools · 186 models · **878 tests** · provider-agnostic LLM routing | Local + Ollama Cloud | [→](https://github.com/atandra2000/AutonomousResearcher) |
| **newsagent** | Autonomous AI research-intelligence agent · daily 15+ source sweep · LLM-reasoned Markdown report · 237 passing tests | Local | [→](https://github.com/atandra2000/news-agent) |

---

## ✍️ Writing

- **[Multi-Head Latent Attention — A Technical Deep-Dive](https://github.com/atandra2000/DeepSeek-v3-Lite/blob/main/MLA.md)** — 643-line reference covering KV-cache math, low-rank compression algebra, the absorption-trick derivation, decoupled RoPE mechanics, and SDPA vs manual attention trade-offs in DeepSeek-V2/V3.
- **[Attention Sinks — StreamingLLM for GPT-OSS](https://github.com/atandra2000/GPT-OSS-Lite/blob/main/ATTENTION_SINKS.md)** — 600-line reference on the learned per-head sink bias, its BF16 numerical-stability story (clamped to [-10, 15]), and its interaction with sliding/full attention alternation.
- **[State-Space Duality — The Mamba-3 Chunkwise SSD](https://github.com/atandra2000/Mamba-3-Lite/blob/main/SSD.md)** — full derivation of the chunkwise SSD algorithm and its equivalence to the naive O(T) recurrence reference.

---

## 🔬 Engineering Themes

- **From-scratch PyTorch** — no Trainer, no Lightning, no accelerate; every layer written by hand
- **Single-GPU feasibility** — BF16, gradient checkpointing, FA2, `channels_last`, fused optimizers
- **Faithful reproductions** — DeepSeek-V3, LLaMA-3, GPT-OSS, Mamba-3, PaliGemma, DCGAN implemented to the paper
- **Novel hybrids** — HyMo (GDN + MLA + MoE + MTP), FaceAgingCycleGAN (AdaIN-conditioned CycleGAN), GPT-OSS-Lite (sliding/full alt + learned sinks + YaRN)
- **Production hygiene** — atomic checkpoints (`.tmp.pt` → `os.rename`), full RNG-state reproducibility, W&B / Comet tracking, CI lint + tests
- **Data pipelines** — resumable download → filter → tokenize → shard → streaming loader, with dedup and document packing; universal 8.0B-token shared pipeline across all LLM projects
- **Post-training & inference** — speculative decoding (MTP-as-draft), Min-SNR loss weighting, EMA, classifier-free guidance
- **Hardware breadth** — MPS / CPU → Kaggle T4 / P100 → A100 80GB → 2× RTX 5090 → RTX 6000 Ada

---

## 🎓 Background

B.Tech, 2024 · Heritage Institute of Technology, Kolkata. Self-taught in deep learning through two years of from-scratch implementation — engineering discipline from infrastructure and constraint work translates directly to memory budgets, distributed training, and reproducible ML systems.

---

## 📫 Connect

[![Portfolio](https://img.shields.io/badge/Portfolio-atandra2000.github.io-1f883d?style=flat-square&logo=google-chrome&logoColor=white)](https://atandra2000.github.io/mycv)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-atandrabharati-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/atandrabharati)
[![GitHub](https://img.shields.io/badge/GitHub-atandra2000-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/atandra2000)
[![W&B](https://img.shields.io/badge/W&B-atandrabharati--self-FCB439?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai/atandrabharati-self)
[![Kaggle](https://img.shields.io/badge/Kaggle-atandrabharati-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://kaggle.com/atandrabharati)
[![Comet](https://img.shields.io/badge/Comet-atandrabharati-2C3E50?style=flat-square)](https://comet.com/atandrabharati)
[![Email](https://img.shields.io/badge/Email-atandra.bharati@gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:atandra.bharati@gmail.com)

---

<div align="center">

<sub>Last updated 2026-07-19 · 17 projects (7 LLM · 8 Vision · 2 Agentic) · Open to remote and on-site roles</sub>

</div>
