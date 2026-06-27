<div align="center">

# Atandra Bharati

**Deep Learning Research Engineer** — building frontier AI architectures from scratch in raw PyTorch.

LLMs · Latent Diffusion · Multimodal · Video Understanding · Agentic ML

`12 from-scratch projects` · `78% memory optimization` · `878-test agentic platform` · `860M-param UNet trained from random init`

</div>

---

## 🎯 Open To

**Deep Learning Research Engineer** · **LLM Engineer** · **GenAI / Diffusion Engineer** · **Agentic ML Engineer**

Remote-friendly · Available worldwide

---

## 🧭 Now

Shipping the **Autonomous ML Research Engineer** platform (15 phases, 23 agents) and exploring **mixture-of-depths routing** for sub-1B parameter LLMs.

---

## 🛠️ Stack

**Languages & ML core**  
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)

**Architectures**  
Transformers · GQA · MLA · RoPE · SwiGLU · RMSNorm · MoE · Gated Delta Net · MTP · Diffusion UNet · VAE · GAN · CycleGAN · ST-GCN · HRNet · SigLIP

**Optimization & numerics**  
BF16 · FP16 · FP8 · Flash Attention 2 · SDPA · `torch.compile` · `channels_last` · Gradient checkpointing · μP scaling · WSD LR · NorMuon · Chunked cross-entropy · Disk-backed token caching · Fused optimizers

**Hardware validated**  
A100 80GB · RTX 5090 (Blackwell) · RTX 6000 Ada · RTX 3090 · P100 · 2× T4

**Tooling**  
HuggingFace · diffusers · tiktoken · W&B · Comet · safetensors · ONNX · TensorRT · FastAPI · pydantic v2 · ChromaDB · Ollama Cloud

---

## 🏆 Highlights

- **78% peak memory reduction (92 GB → 20 GB)** for LLM pretraining via gradient checkpointing, chunked cross-entropy, and disk-backed token caching — enabling 2× batch-size headroom on a single A100 80GB.
- **Training loss 0.0947 at epoch 16** on Stable Diffusion 1.x (860M UNet) trained from random init across a 7-phase curriculum on 2× RTX 5090.
- **~30 FPS inference on RTX 3090** for skeleton-based action recognition, served via ONNX + TensorRT + FastAPI.
- **878 passing tests · 15 cooperating phases · 23 agents · 61 tools · 186 models** in the Autonomous ML Research Engineer platform — full paper-to-conclusions loop with self-repair and provider-agnostic LLM routing.
- **415.6M active / 868.6M stored params** in FusionLLM — a novel hybrid of MLA + Gated Delta Net + MoE + MTP in a 24-layer decoder.
- **643-line technical deep-dive on MLA** (Multi-Head Latent Attention) covering KV-cache math, low-rank compression, the absorption-trick derivation, and decoupled RoPE mechanics.

---

## 📂 Projects

| Domain | Project | Highlight | Hardware | Repo |
|--------|---------|-----------|----------|------|
| **LLM** | **DeepSeek-v3-Lite** (422M) | MLA + AuxLossFreeGate MoE + MTP, end-to-end with absorption-trick inference | A100 80GB | [→](https://github.com/atandra2000/DeepSeek-v3-Lite) |
| **LLM** | **LLaMA-3-Lite** (515M) | GQA · RoPE θ=500K · SwiGLU · RMSNorm · FA2 · chunked CE · **78% memory cut** | A100 80GB | [→](https://github.com/atandra2000/LLaMA-3-Lite) |
| **LLM** | **FusionLLM** (415.6M / 868.6M) | Novel MLA + Gated Delta Net + MoE + MTP hybrid · NorMuon + CautiousAdamW · WSD | A100 80GB | [→](https://github.com/atandra2000/FusionLLM) |
| **LLM** | **GPT-From-Scratch** | 200-line educational GPT-2 with fused QKV; HF weight loading | MPS / CUDA | [→](https://github.com/atandra2000/GPT-From-Scratch) |
| **LLM** | **TranslationLM** (EN→IT) | Encoder–decoder Transformer · loss 6.17 → 2.28 · BLEU/CER/WER | P100 | [→](https://github.com/atandra2000/TranslationLM) |
| **Vision** | **Stable Diffusion 1.x** (860M UNet) | Custom UNet trained from random init · 7 phases · 1.3M+ images · **best loss 0.0947** | 2× RTX 5090 | [→](https://github.com/atandra2000/StableDiffusion) |
| **Vision** | **ActionRecognition** (120 cls) | HRNet pose + Two-Stream CTR-GCN · **~30 FPS** · ONNX + TensorRT | RTX 3090 | [→](https://github.com/atandra2000/ActionRecognition) |
| **Vision** | **FaceAgingCycleGAN** (256²) | Per-layer AdaIN conditioning · 3-scale PatchGAN · LSGAN + R1 GP | RTX 6000 Ada | [→](https://github.com/atandra2000/FaceAgingCycleGAN) |
| **Vision** | **FaceGenerationVAE** (β-VAE) | 50 epochs · recon MSE 0.0152 · linear KL annealing · bilinear-upsample decoder | P100 | [→](https://github.com/atandra2000/FaceGenerationVAE) |
| **Vision** | **DCGAN-Face-Generation** | 50 epochs · 202K CelebA · D loss → ln 2 ≈ 0.693 equilibrium | 2× T4 | [→](https://github.com/atandra2000/DCGAN-Face-Generation) |
| **Multimodal** | **VisionLangModel** (PaliGemma-style) | SigLIP ViT + Gemma decoder + linear projector · zero pretrained weights | P100 | [→](https://github.com/atandra2000/VisionLanguageModel) |
| **Agentic** | **Autonomous ML Research Engineer** | 15-phase multi-agent platform · paper → plan → patch → train → evaluate → report | Local + Ollama Cloud | [→](https://github.com/atandra2000/AutonomousResearcher) |

---

## ✍️ Writing

- **[Multi-Head Latent Attention — A Technical Deep-Dive](https://github.com/atandra2000/DeepSeek-v3-Lite/blob/main/MLA.md)** — 643-line reference covering KV-cache math, low-rank compression algebra, the absorption-trick derivation, decoupled RoPE mechanics, and SDPA vs manual attention trade-offs in DeepSeek-V2/V3.

---

## 🔬 Engineering Themes

- **From-scratch PyTorch** — no Trainer, no Lightning, no accelerate; every layer written by hand
- **Single-GPU feasibility** — BF16, gradient checkpointing, FA2, `channels_last`, fused optimizers
- **Faithful reproductions** — DeepSeek-V3, LLaMA-3, PaliGemma, DCGAN implemented to the paper
- **Novel hybrids** — FusionLLM (MLA + GDN + MoE + MTP), FaceAgingCycleGAN (AdaIN-conditioned CycleGAN)
- **Production hygiene** — atomic checkpoints (`.tmp.pt` → `os.rename`), full RNG-state reproducibility, W&B / Comet tracking, CI lint + tests
- **Data pipelines** — resumable download → filter → tokenize → shard → streaming loader, with dedup and document packing
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

<sub>Last updated 2026-06-27 · Open to remote and on-site roles</sub>

</div>
