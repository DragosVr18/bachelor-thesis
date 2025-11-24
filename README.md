This repository contains the implementation, modifications, and experiments conducted for developing a lightweight version of MultiHMR, a transformer-based model for multi-person 3D human mesh recovery.

🔗 Original MultiHMR repository:
https://github.com/naver/multi-hmr

⸻

### Project Overview

This project is a heavily modified fork of MultiHMR. The goal is to reduce model size and computation cost while maintaining competitive prediction quality—making MultiHMR more suitable for resource-limited devices and real-time applications.

The practical work includes:
	•	Replacing MultiHMR’s ViT-S encoder with TinyViT-5M
	•	Reducing and re-designing the cross-attention decoder (HPH)
	•	Implementing teacher–student distillation
	•	Training and fine-tuning on BEDLAM, 3DPW, and AGORA
	•	Building a working FastAPI + React demo application for inference

⸻

### Model Modifications

1. Lightweight Cross-Attention Decoder
	•	Original embedding: 1024 → reduced to 256–384
	•	Attention heads: 32 → 16–24
	•	Result: millions fewer parameters with modest accuracy drop
	•	Includes distillation from the original MultiHMR model

2. TinyViT Encoder Replacement
	•	Original ViT-S (21M params) replaced with TinyViT-5M
	•	Required joint training (encoder + decoder) for stable convergence
	•	Final model size: ~14.5M params (~50% reduction from original)

⸻

### Training & Evaluation

Experiments were performed on:
	•	BEDLAM (subset of 77k images)
	•	3DPW (fine-tuning for real-world data)
	•	AGORA (crowded, occluded scenes)

Metrics:
	•	MPJPE / PA-MPJPE
	•	PVE / PA-PVE
	•	F1 score for multi-person detection

Key findings:
	•	Lightweight model performs well in close-range and few-person scenes
	•	Larger performance drop in crowded scenes (AGORA) due to encoder limits
	•	Distillation helps only for decoder reduction, not when replacing encoder

⸻

### Demo Application

A functional web application is included:
	•	Built with FastAPI (backend) and React (frontend)
	•	Accepts image upload or webcam frames
	•	Outputs 3D SMPL-X mesh rendered directly in the browser
