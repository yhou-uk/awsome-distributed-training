# Troubleshooting Guide

This guide covers common issues and solutions when running the QLoRA fine-tuning tutorial.

## Table of Contents

1. [EKS Cluster Issues](#eks-cluster-issues)
2. [GPU and CUDA Issues](#gpu-and-cuda-issues)
3. [DeepSpeed and Multi-GPU Issues](#deepspeed-and-multi-gpu-issues)
4. [Training Issues](#training-issues)
5. [Docker Issues](#docker-issues)
6. [Model Loading Issues](#model-loading-issues)
7. [Inference Issues](#inference-issues)

---

## EKS Cluster Issues

### Cluster Creation Fails

**Symptom**: `eksctl create cluster` fails with errors

**Solutions**:

1. Check IAM permissions:
   ```bash
   aws sts get-caller-identity
   ```

2. Check CloudFormation for details:
   ```bash
   aws cloudformation describe-stack-events \
       --stack-name eksctl-qwen3-qlora-cluster-cluster \
       --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`]'
   ```

3. Try a different availability zone. The g5.12xlarge instance may not be
   available in all AZs. Check capacity:
   ```bash
   aws ec2 describe-instance-type-offerings \
       --location-type availability-zone \
       --filters Name=instance-type,Values=g5.12xlarge \
       --region us-east-1 \
       --query 'InstanceTypeOfferings[].Location'
   ```

### Cannot Connect to Cluster

**Symptom**: `kubectl` commands fail with connection errors

**Solutions**:

1. Update kubeconfig:
   ```bash
   aws eks update-kubeconfig --region us-east-1 --name qwen3-qlora-cluster
   ```

2. Check context:
   ```bash
   kubectl config current-context
   kubectl config get-contexts
   ```

3. Verify cluster is active:
   ```bash
   aws eks describe-cluster --name qwen3-qlora-cluster --query 'cluster.status'
   ```

### GPU Nodes Not Available

**Symptom**: GPU nodes show 0 or pending

**Solutions**:

1. Check node group status:
   ```bash
   eksctl get nodegroup --cluster qwen3-qlora-cluster
   ```

2. Scale up GPU nodes:
   ```bash
   eksctl scale nodegroup --cluster qwen3-qlora-cluster \
       --name gpu-nodes --nodes 1 --region us-east-1
   ```

3. Check for capacity issues in AWS Console (EC2 > Auto Scaling Groups).
   g5.12xlarge may have limited availability in some AZs.

---

## GPU and CUDA Issues

### No GPU Detected

**Symptom**: `torch.cuda.is_available()` returns `False`

**Solutions**:

1. Check NVIDIA driver:
   ```bash
   nvidia-smi
   ```

2. Check CUDA version:
   ```bash
   nvcc --version
   ```

3. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

### CUDA Out of Memory (Single GPU)

**Symptom**: `CUDA out of memory` or `CUDA error: an illegal memory access` during training on a single GPU

This is the most common issue with single-GPU training. The Qwen3-8B QLoRA
setup uses ~20-23GB on a 24GB A10G, leaving minimal headroom for memory spikes
during checkpoint saves.

**Solutions**:

1. **Switch to multi-GPU with DeepSpeed ZeRO-2** (recommended):
   ```yaml
   # In configs/training_config.yaml
   parallel:
     strategy: "deepspeed_zero2"
   ```
   Then launch with `torchrun --nproc_per_node=4` on a g5.12xlarge.

2. Reduce sequence length:
   ```yaml
   model:
     max_seq_length: 1024  # Reduced from 1536
   ```

3. Reduce batch size:
   ```yaml
   training:
     per_device_train_batch_size: 1
     gradient_accumulation_steps: 8  # Increase to compensate
   ```

4. Enable memory-friendly CUDA allocator:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
   ```

5. Enable gradient checkpointing (enabled by default):
   ```yaml
   parallel:
     gradient_checkpointing: true
   ```

### CUDA Out of Memory (Multi-GPU with ZeRO-2)

**Symptom**: OOM even with 4 GPUs and DeepSpeed ZeRO-2

ZeRO-2 shards optimizer states and gradients, but each GPU still holds the
full model parameters and activations. If you still OOM:

1. Reduce `max_seq_length` (e.g., 1536 -> 1024). Activation memory scales
   quadratically with sequence length in attention.

2. Switch to DeepSpeed ZeRO-3 which also shards model parameters:
   ```yaml
   parallel:
     strategy: "deepspeed_zero3"
   ```

3. Check for memory fragmentation:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
   ```

### NVIDIA Device Plugin Not Working

**Symptom**: Pods stuck in `Pending` with GPU requests

**Solutions**:

1. Reinstall device plugin:
   ```bash
   kubectl delete daemonset nvidia-device-plugin-daemonset -n kube-system
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml
   ```

2. Check device plugin logs:
   ```bash
   kubectl logs -n kube-system -l name=nvidia-device-plugin-ds
   ```

3. Verify GPU node labels:
   ```bash
   kubectl get nodes -l nvidia.com/gpu=true
   ```

---

## DeepSpeed and Multi-GPU Issues

### NCCL Communication Errors

**Symptom**: `NCCL error`, `RuntimeError: NCCL communicator was aborted`, or
training hangs during the first all-reduce operation.

**Solutions**:

1. Check NCCL debug output:
   ```bash
   # Set in kubernetes/training-job.yaml env vars:
   NCCL_DEBUG=INFO
   NCCL_SOCKET_IFNAME=eth0
   ```

2. Verify all GPUs are visible within the pod:
   ```bash
   kubectl exec -it <pod-name> -n ml-training -- nvidia-smi
   ```
   You should see 4 GPUs listed.

3. Increase shared memory. NCCL uses `/dev/shm` for inter-GPU communication:
   ```yaml
   # In training-job.yaml (already configured):
   volumes:
     - name: shm
       emptyDir:
         medium: Memory
         sizeLimit: "32Gi"
   ```

### DeepSpeed Checkpoint Incompatibility

**Symptom**: `ValueError: Can't find a valid checkpoint at /workspace/outputs/checkpoint-XXX`

This happens when trying to resume a DeepSpeed checkpoint from a non-DeepSpeed
run, or vice versa. DeepSpeed checkpoints contain `global_step*` directories;
vanilla HF Trainer checkpoints do not.

**Solutions**:

1. The training script auto-detects this mismatch and skips incompatible
   checkpoints. If you see this warning, training will restart from scratch.

2. To manually check checkpoint format:
   ```bash
   ls /workspace/outputs/checkpoint-*/
   # DeepSpeed: contains global_step*/ directories
   # Vanilla HF: contains pytorch_model.bin or model.safetensors
   ```

3. To force a fresh start, delete old checkpoints:
   ```bash
   kubectl exec -it <pod-name> -n ml-training -- rm -rf /workspace/outputs/checkpoint-*
   ```

### torchrun Errors

**Symptom**: `torchrun` fails to launch or processes crash immediately

**Solutions**:

1. Verify `--nproc_per_node` matches the number of GPUs:
   ```yaml
   command:
     - torchrun
     - "--nproc_per_node=4"  # Must match nvidia.com/gpu resource request
     - "--master_port=29500"
     - "/app/src/train.py"
   ```

2. Check that `CUDA_VISIBLE_DEVICES` is NOT set. torchrun manages GPU
   assignment via `LOCAL_RANK`.

3. Verify the `--master_port` is not in use by another process.

---

## Training Issues

### Loss Not Decreasing

**Symptom**: Training loss stays flat or increases

**Solutions**:

1. Lower learning rate:
   ```yaml
   training:
     learning_rate: 1.0e-4  # Reduce from 2e-4
   ```

2. Increase warmup:
   ```yaml
   training:
     warmup_ratio: 0.1  # Increase from 0.03
   ```

3. Check data formatting:
   ```python
   # Inspect a sample
   from datasets import load_dataset
   ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
   print(ds['train'][0])
   ```

### Training Crashes Mid-Run

**Symptom**: Training stops unexpectedly

**Solutions**:

1. Check for OOM:
   ```bash
   dmesg | grep -i "killed process"
   ```

2. Auto-resume is enabled by default. The training script uses
   `--resume_from_checkpoint=auto` which finds the latest valid checkpoint:
   ```bash
   # Check if checkpoints exist
   kubectl exec -it <pod-name> -n ml-training -- ls /workspace/outputs/
   ```

3. For EKS pods, check shared memory is large enough:
   ```yaml
   # In kubernetes/training-job.yaml (already configured at 32Gi):
   volumes:
     - name: shm
       emptyDir:
         medium: Memory
         sizeLimit: "32Gi"
   ```

### Slow Training

**Symptom**: Training is much slower than expected

**Solutions**:

1. Verify all GPUs are being used:
   ```python
   import torch
   print(f"GPUs available: {torch.cuda.device_count()}")
   for i in range(torch.cuda.device_count()):
       print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
   ```

2. Check for CPU bottleneck:
   ```yaml
   # Increase DataLoader workers
   data:
     num_workers: 8
   ```

3. Disable debugging:
   ```bash
   export CUDA_LAUNCH_BLOCKING=0
   ```

4. Check NCCL performance. Slow all-reduce can bottleneck multi-GPU training:
   ```bash
   export NCCL_DEBUG=INFO  # Check for warnings in logs
   ```

---

## Docker Issues

### Docker Build Fails

**Symptom**: `docker build` fails

**Solutions**:

1. Check Docker is running:
   ```bash
   docker info
   ```

2. Clear Docker cache:
   ```bash
   docker system prune -a
   ```

3. Build with verbose output:
   ```bash
   docker build --progress=plain -t qwen3-qlora-training:test -f docker/Dockerfile .
   ```

### ECR Push Fails

**Symptom**: Cannot push image to ECR

**Solutions**:

1. Re-authenticate:
   ```bash
   aws ecr get-login-password --region us-east-1 | \
       docker login --username AWS --password-stdin \
       $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com
   ```

2. Check ECR repository exists:
   ```bash
   aws ecr describe-repositories --repository-names qwen3-qlora-training
   ```

3. Check IAM permissions for ECR access

---

## Model Loading Issues

### Transformers Version Error

**Symptom**: `KeyError: 'qwen3'` when loading model

**Solution**: Upgrade transformers:
```bash
pip install transformers>=4.51.0
```

### Model Download Fails

**Symptom**: Network error when downloading from Hugging Face

**Solutions**:

1. Use Hugging Face token:
   ```bash
   huggingface-cli login
   ```

2. Set HF_HOME for caching:
   ```bash
   export HF_HOME=/path/to/cache
   ```

3. Download manually:
   ```bash
   git lfs install
   git clone https://huggingface.co/Qwen/Qwen3-8B
   ```

### bitsandbytes Errors

**Symptom**: `bitsandbytes` import fails

**Solutions**:

1. Reinstall with CUDA:
   ```bash
   pip uninstall bitsandbytes
   pip install bitsandbytes>=0.42.0
   ```

2. Check CUDA compatibility:
   ```python
   import bitsandbytes as bnb
   print(bnb.cuda_available)
   ```

3. For specific CUDA versions:
   ```bash
   # For CUDA 11.8
   pip install bitsandbytes --prefer-binary --extra-index-url=https://pypi.nvidia.com
   ```

---

## Inference Issues

### Full-Precision Model OOM

**Symptom**: Loading the merged bf16 model for inference runs out of memory

The merged full-precision model requires ~16GB VRAM. If your GPU has less:

1. Use the 4-bit quantized loading (Section 11 in the inference notebook)
2. Use CPU offloading:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen3-8B",
       torch_dtype=torch.bfloat16,
       device_map="auto",  # Will offload layers to CPU if needed
   )
   ```

### LoRA Adapter Not Found

**Symptom**: `OSError: outputs/final_model is not a valid LoRA adapter`

**Solutions**:

1. Verify the adapter files exist:
   ```bash
   ls outputs/final_model/
   # Should contain: adapter_config.json, adapter_model.safetensors
   ```

2. If using the EKS-trained model, copy it from the PVC first:
   ```bash
   # Create a temp pod attached to the PVC, then:
   kubectl cp ml-training/<pod-name>:/workspace/outputs/final_model ./outputs/final_model
   ```

---

## Getting Help

If you're still stuck:

1. Check the logs:
   ```bash
   # EKS pod logs
   kubectl logs job/qwen3-qlora-training -n ml-training

   # Previous pod attempt logs (if restarted)
   kubectl logs job/qwen3-qlora-training -n ml-training --previous
   ```

2. Create an issue on GitHub with:
   - Full error message
   - Environment details (GPU, CUDA version, Python version)
   - Steps to reproduce

3. Common commands for debugging:
   ```bash
   # System info
   nvidia-smi
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   python -c "import transformers; print(transformers.__version__)"

   # EKS info
   kubectl get pods -n ml-training
   kubectl describe pod <pod-name> -n ml-training
   kubectl get events -n ml-training --sort-by='.lastTimestamp'
   ```
