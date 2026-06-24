# Atandra Bharati

**Deep Learning Research Engineer** rebuilding frontier AI architectures from scratch — LLMs, latent diffusion, multimodal, video understanding. PyTorch-first, single-GPU heroics, paper-faithful reproductions.

---

### 🧭 Open to roles

Applied ML · ML Research · Research Engineer · GenAI Engineering. Remote-friendly; available worldwide.

### 🛠️ Core stack

**Languages & core ML** &nbsp;
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)

**Architectures** &nbsp;
Transformers · GQA · MLA · RoPE · SwiGLU · RMSNorm · MoE · Gated Delta Net · MTP · Diffusion UNet · VAE · GAN · CycleGAN · ST-GCN

**Optimization & numerics** &nbsp;
BF16 · Flash Attention 2 · `torch.compile` · Gradient checkpointing · μP scaling · WSD LR · Chunked cross-entropy · Disk-backed token caching

**Hardware validated** &nbsp;
A100 80GB · RTX 5090 · RTX 6000 Ada · RTX 3090 · P100 · T4 (2×)

### 🔭 Now

Shipping the **Autonomous ML Research Engineer** platform (15 phases, 23 agents) and exploring a paper on **mixture-of-depths routing** for sub-1B parameter LLMs.

---

## Highlights

- **78% peak memory reduction (92 GB → 20 GB)** for LLM pretraining via gradient checkpointing, chunked cross-entropy, and disk-backed token caching — enabling 2× batch-size headroom on a single A100 80GB.
- **Training loss 0.0947 at epoch 16** on Stable Diffusion 1.x (860M UNet) trained from scratch across a 7-phase curriculum on 2× RTX 5090.
- **878 passing tests, 15 cooperating phases, 23 agents, 61 tools, 186 models** in the Autonomous ML Research Engineer platform — a full research loop from paper to conclusions, with self-repair and provider-agnostic LLM routing.
- **12 end-to-end projects** spanning LLMs, generative vision, multimodal AI, and video — every project engineered for single-GPU feasibility.

---

## Projects

| Category | Project | Highlight | Stack / hardware | Repo |
|----------|---------|-----------|------------------|------|
| Architecture | **DeepSeek-v3-Lite** (422M) | MLA + aux-loss-free MoE + MTP, end-to-end with inference absorption | PyTorch · μP · 8.4B-token Chinchilla recipe | [→](https://github.com/atandra2000/DeepSeek-v3-Lite) |
| Architecture | **LLaMA-3-Lite** (515M) | GQA · RoPE · fused SwiGLU · RMSNorm · Flash-Attn 2 · chunked CE | PyTorch · BF16 · A100 80GB | [→](https://github.com/atandra2000/LLaMA-3-Lite) |
| Architecture | **FusionLLM** (415.6M active / 868.6M stored) | MLA + Gated Delta Net + MoE + MTP in a 24-layer hybrid | PyTorch · NorMuon + CautiousAdamW · WSD + μP | [→](https://github.com/atandra2000/FusionLLM) |
| Generative vision | **Stable Diffusion 1.x** (860M UNet) | Best loss **0.0947** at epoch 16; 42-epoch run | PyTorch · BF16 · 2× RTX 5090 | [→](https://github.com/atandra2000/StableDiffusion) |
| Generative vision | **FaceAgingCycleGAN** (AdaIN-conditioned) | 31 epochs on IMDB-WIKI; per-layer age conditioning, 3-scale PatchGAN | PyTorch · RTX 6000 Ada | [→](https://github.com/atandra2000/FaceAgingCycleGAN) |
| Generative vision | **FaceGenerationVAE** (β-VAE) | 50 epochs on CelebA; recon MSE 0.0152, KL annealing 0→1 | PyTorch · bilinear-upsample decoder | [→](https://github.com/atandra2000/FaceGenerationVAE) |
| Generative vision | **DCGAN-Face-Generation** | 50 epochs on 202k CelebA; D loss → ln 2 ≈ 0.693 equilibrium | PyTorch · 2× T4 | [→](https://github.com/atandra2000/DCGAN-Face-Generation) |
| Multimodal | **VisionLangModel** (PaliGemma-style) | Trained end-to-end on COCO 2014 captions; zero pre-trained weights | PyTorch · P100 | [→](https://github.com/atandra2000/VisionLanguageModel) |
| NLP | **TranslationLM** (EN→IT seq2seq) | 20 epochs on OPUS Books; cross-attention visualizations, custom SentencePiece BPE | PyTorch · T4 | [→](https://github.com/atandra2000/TranslationLM) |
| Foundations | **GPT-From-Scratch** | 200-line educational GPT-2, trained on Tiny Shakespeare | PyTorch | [→](https://github.com/atandra2000/GPT-From-Scratch) |
| Agentic / research infra | **Autonomous ML Research Engineer** | 15-phase multi-agent platform: paper → plan → patch → train → evaluate → iterate → report | PyTorch · Ollama Cloud · multi-agent · 878 tests | [→](https://github.com/atandra2000/AutonomousResearcher) |
| In progress | **ActionRecognition** (ST-GCN) | Pose + ST-GCN pipeline ready; NTU RGB+D 120 benchmark pending | PyTorch | [→](https://github.com/atandra2000/ActionRecognition) |

---

## Writing

- **"Multi-Head Latent Attention — A Technical Deep-Dive"** — 643-line reference covering KV-cache math, low-rank compression algebra, the absorption-trick derivation, decoupled RoPE mechanics, and SDPA vs manual attention path trade-offs in DeepSeek-V2/V3. ([read](https://github.com/atandra2000/DeepSeek-v3-Lite/blob/main/MLA.md))

---

## Connect

[![Portfolio](https://img.shields.io/badge/Portfolio-atandra2000.github.io%2Fmycv-0A0A0A?style=flat-square&logo=google-chrome&logoColor=white)](https://atandra2000.github.io/mycv)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-atandrabharati-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/atandrabharati)
[![Weights & Biases](https://img.shields.io/badge/W%26B-atandrabharati--self-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai/atandrabharati-self)
[![Kaggle](https://img.shields.io/badge/Kaggle-atandrabharati-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://kaggle.com/atandrabharati)
[![Comet](https://img.shields.io/badge/Comet-atandrabharati-2C3E50?style=flat-square)](https://comet.com/atandrabharati)