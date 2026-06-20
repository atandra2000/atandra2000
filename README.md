# Atandra Bharati

**Deep learning research engineer** rebuilding frontier AI architectures from scratch in PyTorch — LLMs, latent diffusion, multimodal, video.

## What I do

- **11 from-scratch projects** spanning LLMs (LLaMA-3, DeepSeek-V3, FusionLLM, GPT-2, TranslationLM), generative vision (Stable Diffusion, DCGAN, VAE, CycleGAN), multimodal (PaliGemma-style VLM), and video understanding (ST-GCN action recognition).
- **Single-GPU feasibility**: engineered to run on A100, RTX 5090, RTX 6000 Ada, RTX 3090, P100, T4 with BF16, Flash Attention 2, gradient checkpointing, and torch.compile.
- **Faithful paper reproductions**: MLA with absorption trick, aux-loss-free MoE, Multi-Token Prediction, Min-SNR, KL annealing, AdaIN conditioning.

## Verified builds

| Project | Status | Link |
|---------|--------|------|
| Stable Diffusion 1.x (860M UNet) | Trained 42 epochs on 2x RTX 5090 | [repo](https://github.com/atandra2000/StableDiffusion) |
| VisionLanguageModel (PaliGemma-style) | Trained end-to-end on COCO captions | [repo](https://github.com/atandra2000/VisionLanguageModel) |
| FaceAgingCycleGAN (AdaIN) | 31 epochs on IMDB-WIKI | [repo](https://github.com/atandra2000/FaceAgingCycleGAN) |
| FaceGenerationVAE (beta-VAE) | 50 epochs on CelebA | [repo](https://github.com/atandra2000/FaceGenerationVAE) |
| DCGAN-Face-Generation | 50 epochs on CelebA | [repo](https://github.com/atandra2000/DCGAN-Face-Generation) |
| GPT-From-Scratch | Trained on Tiny Shakespeare | [repo](https://github.com/atandra2000/GPT-From-Scratch) |
| TranslationLM (EN->IT) | 20 epochs on OPUS Books | [repo](https://github.com/atandra2000/TranslationLM) |

## In-progress architectures

| Project | Status | Link |
|---------|--------|------|
| DeepSeek-V3-Lite (MLA + MoE + MTP) | Architecture and pipeline ready; pre-training not started | [repo](https://github.com/atandra2000/DeepSeek-v3-Lite) |
| LLaMA-3-Lite (515M) | Architecture and pipeline ready; pre-training not started | [repo](https://github.com/atandra2000/LLaMA-3-Lite) |
| FusionLLM (MLA + GDN + MoE + MTP) | Framework and tests ready; pre-training not started | [repo](https://github.com/atandra2000/FusionLLM) |
| ActionRecognition (ST-GCN) | Pipeline and tests ready; NTU benchmark not started | [repo](https://github.com/atandra2000/ActionRecognition) |

## Links

- Portfolio site: [atandra2000.github.io/mycv](https://atandra2000.github.io/mycv)
- LinkedIn: [linkedin.com/in/atandrabharati](https://www.linkedin.com/in/atandrabharati)
- Weights and Biases: [wandb.ai/atandrabharati-self](https://wandb.ai/atandrabharati-self)
- Kaggle: [kaggle.com/atandrabharati](https://kaggle.com/atandrabharati)
- Comet: [comet.com/atandrabharati](https://comet.com/atandrabharati)
