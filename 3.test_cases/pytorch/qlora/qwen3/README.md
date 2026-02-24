# QLoRA Fine-tuning: Qwen3-8B on Amazon EKS

Fine-tune [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) using QLoRA (4-bit quantization + LoRA adapters) with DDP and DeepSpeed ZeRO-2 on Amazon EKS.

## What This Does

- Loads Qwen3-8B in 4-bit precision using BitsAndBytes NF4 quantization
- Adds LoRA adapters (rank 16) to all attention and MLP projection layers
- Trains on [medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) (19.7k English samples)
- Uses DDP + DeepSpeed ZeRO-2 to shard optimizer states and gradients across 4 GPUs
- Runs on a single g5.12xlarge node (4x NVIDIA A10G, 24GB each)

## Why DDP + DeepSpeed ZeRO-2

Training Qwen3-8B with QLoRA on a single A10G (24GB) hits ~99% VRAM utilization and crashes at checkpoint-save points. With 4 GPUs and ZeRO-2, per-GPU memory drops from ~23GB to ~15GB by sharding optimizer states and gradients:

| Component | Single GPU | 4x GPU + ZeRO-2 |
|-----------|-----------|------------------|
| Base model (4-bit) | ~5 GB | ~5 GB |
| LoRA adapters | ~0.1 GB | ~0.1 GB |
| Activations | ~8 GB | ~5 GB |
| Optimizer states | ~3 GB | ~0.75 GB |
| Gradients | ~2 GB | ~0.5 GB |
| **Total** | **~20 GB** | **~15 GB** |

## Prerequisites

- AWS account with permissions for EKS, ECR, and EC2 (g5 instances)
- AWS CLI, eksctl, kubectl, Docker installed
- [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) installed on the cluster (installed automatically by `0.setup-cluster.sh`)

## Quick Start

```bash
# 0. Create EKS cluster with GPU nodes (one-time setup)
./0.setup-cluster.sh

# 1. Build and push the Docker image
./1.build-image.sh

# 2. Deploy the training job
export IMAGE=<account>.dkr.ecr.<region>.amazonaws.com/qwen3-qlora-training:latest
./2.deploy-training.sh

# 3. Monitor
kubectl logs -f job/qwen3-qlora-training -n ml-training

# 4. Cleanup
./3.cleanup.sh
```

See [kubernetes/README.md](kubernetes/README.md) for detailed setup instructions and [docs/](docs/) for additional documentation.

## Project Structure

```
qwen3/
├── README.md                           # This file
├── Dockerfile                          # Training container image
├── requirements.txt                    # Python dependencies
├── src/
│   ├── train.py                        # Main training script (DDP + DeepSpeed)
│   ├── config.py                       # Configuration dataclasses
│   ├── model_setup.py                  # QLoRA model loading
│   └── data_preparation.py            # Dataset formatting and tokenization
├── configs/
│   ├── training_config.yaml            # All hyperparameters
│   ├── deepspeed_zero2.json            # DeepSpeed ZeRO-2 config (default)
│   └── deepspeed_zero3.json            # DeepSpeed ZeRO-3 config (fallback)
├── kubernetes/
│   ├── README.md                       # EKS deployment instructions
│   ├── eks-cluster.yaml                # eksctl cluster config
│   └── qwen3_8b-qlora.yaml            # Kubernetes Job manifest
├── notebooks/
│   ├── qwen3_qlora_finetuning.ipynb   # Step-by-step training notebook
│   └── inference_demo.ipynb           # Inference with fine-tuned model
├── docs/
│   ├── EKS_SETUP.md                   # Detailed EKS setup guide
│   ├── QLORA_EXPLAINED.md             # QLoRA concepts and architecture
│   └── TROUBLESHOOTING.md            # Common issues and solutions
├── 0.setup-cluster.sh                  # Create EKS cluster + GPU nodes
├── 1.build-image.sh                    # Build + push Docker image to ECR
├── 2.deploy-training.sh                # Deploy training job to EKS
└── 3.cleanup.sh                        # Delete training resources
```

## Configuration Reference

All hyperparameters are in `configs/training_config.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch size | 1 per GPU | Per-device batch size |
| Gradient accumulation | 4 | Steps before weight update |
| Effective batch size | 16 | 1 x 4 GPUs x 4 accumulation |
| Learning rate | 2e-4 | Peak LR (cosine schedule) |
| Max sequence length | 1536 | Tokens per sample |
| Epochs | 3 | Training epochs |
| Optimizer | paged_adamw_8bit | Memory-efficient optimizer |
| Precision | bfloat16 | Mixed precision |
| LoRA rank | 16 | Low-rank adapter dimension |
| LoRA alpha | 32 | Scaling factor (2x rank) |
| Parallelism | DDP + DeepSpeed ZeRO-2 | Optimizer/gradient sharding |

### QLoRA Settings

```yaml
qlora:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
  lora_r: 16
  lora_alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

### Switching to ZeRO-3

If ZeRO-2 still runs out of memory (e.g., with longer sequences or larger batch sizes), switch to ZeRO-3:

```yaml
# In configs/training_config.yaml
parallel:
  strategy: "deepspeed_zero3"
```

## Troubleshooting

### CUDA Out of Memory

1. Reduce `max_seq_length` (e.g., 1536 -> 1024)
2. Ensure `per_device_train_batch_size: 1`
3. Switch to `deepspeed_zero3` for full parameter sharding
4. Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True`

### NCCL Communication Errors

1. Verify all 4 GPUs are visible: `kubectl exec -it <pod> -- nvidia-smi`
2. Check that `/dev/shm` is mounted with sufficient size (32Gi)
3. Set `NCCL_DEBUG=INFO` to diagnose (already set in the manifest)

### DeepSpeed Checkpoint Incompatibility

The training script auto-detects checkpoint format mismatches. If switching between DeepSpeed and non-DeepSpeed runs, incompatible checkpoints are skipped and training starts fresh.

### Pod Stuck in Pending

1. Check GPU node is scaled up: `eksctl get nodegroup --cluster qwen3-qlora-cluster`
2. Verify NVIDIA device plugin is running: `kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds`
3. Check node taints/tolerations match

## Checkpoint Resume

The training script supports automatic checkpoint resume:
- Uses `--resume_from_checkpoint=auto` by default
- Finds the latest valid checkpoint in the output directory
- Validates checkpoint format compatibility with current DeepSpeed strategy
- Safe for pod restarts (checkpoints are on a PersistentVolumeClaim)
