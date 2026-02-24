# Amazon EKS Setup Guide

This guide provides detailed instructions for setting up an Amazon EKS cluster for GPU-based machine learning workloads.

## Prerequisites

### Required Tools

1. **AWS CLI** (v2)
   ```bash
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install
   aws --version
   ```

2. **eksctl** (latest)
   ```bash
   curl --silent --location "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
   sudo mv /tmp/eksctl /usr/local/bin
   eksctl version
   ```

3. **kubectl** (compatible with your EKS version)
   ```bash
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
   kubectl version --client
   ```

4. **Docker** (for building images)
   ```bash
   # Install Docker
   sudo yum install -y docker  # Amazon Linux
   # or
   sudo apt-get install -y docker.io  # Ubuntu

   sudo systemctl start docker
   sudo usermod -aG docker $USER
   ```

### AWS IAM Permissions

Your IAM user/role needs these policies:
- `AmazonEKSClusterPolicy`
- `AmazonEKSServicePolicy`
- `AmazonEC2FullAccess`
- `IAMFullAccess`
- `AWSCloudFormationFullAccess`
- `AmazonEKS_CNI_Policy`
- `AmazonECR-Full`

### Configure AWS CLI

```bash
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region: us-east-1
# - Output format: json
```

## Cluster Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        VPC (10.0.0.0/16)                      │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐│
│  │                    EKS Control Plane                      ││
│  │                    (AWS Managed)                           ││
│  └──────────────────────────────────────────────────────────┘│
│                              │                                │
│         ┌────────────────────┴────────────────────┐          │
│         │                                          │          │
│  ┌──────▼──────┐                          ┌───────▼────────┐ │
│  │ System Node  │                          │   GPU Node     │ │
│  │ (m5.large)   │                          │ (g5.12xlarge)  │ │
│  │              │                          │                │ │
│  │ - CoreDNS    │                          │ - 4x A10G GPUs │ │
│  │ - VPC-CNI    │                          │ - 24GB each    │ │
│  │ - Metrics    │                          │ - 96GB total   │ │
│  │              │                          │ - DDP+ZeRO-2   │ │
│  └──────────────┘                          └────────────────┘ │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐│
│  │                    EBS Storage (gp3)                       ││
│  │               Model Checkpoints & Cache                    ││
│  └──────────────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────────────┘
```

## Step-by-Step Setup

### Step 1: Create the Cluster

The cluster configuration is in `eks/cluster-config.yaml`. Key settings:

```yaml
metadata:
  name: qwen3-qlora-cluster
  region: us-east-1
  version: "1.29"

managedNodeGroups:
  # System node for K8s components
  - name: system-nodes
    instanceType: m5.large
    desiredCapacity: 1

  # GPU node for multi-GPU training
  - name: gpu-nodes
    instanceType: g5.12xlarge  # 4x NVIDIA A10G (96GB total)
    desiredCapacity: 1         # 1 node with 4 GPUs
    minSize: 0                 # Scale to 0 when idle
    maxSize: 1
```

Create the cluster:

```bash
eksctl create cluster -f eks/cluster-config.yaml
```

This takes 15-20 minutes. You can monitor progress in the AWS Console:
- CloudFormation: https://console.aws.amazon.com/cloudformation
- EKS: https://console.aws.amazon.com/eks

### Step 2: Install NVIDIA Device Plugin

This allows Kubernetes to schedule GPU workloads:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml
```

Verify GPUs are available:

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu
```

Expected output:
```
NAME                                           GPU
ip-10-0-xx-xx.us-east-1.compute.internal      <none>
ip-10-0-xx-xx.us-east-1.compute.internal      <none>
ip-10-0-xx-xx.us-east-1.compute.internal      4
```

The GPU node should report `4` GPUs (the 4x A10G GPUs in the g5.12xlarge instance).

### Step 3: Set Up Storage

Create a storage class for persistent volumes:

```bash
kubectl apply -f kubernetes/storageclass.yaml
```

Or manually:

```bash
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3-storage
provisioner: ebs.csi.aws.com
volumeBindingMode: WaitForFirstConsumer
parameters:
  type: gp3
  fsType: ext4
  encrypted: "true"
EOF
```

### Step 4: Create ECR Repository

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1

aws ecr create-repository \
    --repository-name qwen3-qlora-training \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true
```

### Step 5: Verify Cluster

```bash
# Check nodes
kubectl get nodes

# Check system pods
kubectl get pods -n kube-system

# Check GPU allocation on the GPU node
kubectl describe node -l node-type=gpu | grep -A5 "Allocatable:"
```

## GPU Instance Selection

This project uses **g5.12xlarge** (4x A10G GPUs in a single node) for multi-GPU
training with DDP + DeepSpeed ZeRO-2. All 4 GPUs communicate via NVLink within
the same machine, so there is no cross-node network overhead.


> **Why g5.12xlarge?** A single A10G (24GB) runs at ~99% VRAM utilization for
> Qwen3-8B QLoRA, leading to CUDA OOM crashes at checkpoint-save points. The
> 4-GPU setup with DeepSpeed ZeRO-2 reduces per-GPU memory to ~15GB by sharding
> optimizer states and gradients across GPUs.

## Cost Optimization

### Scale GPU Nodes to Zero

When not training, scale the GPU node group to 0 to save cost:

```bash
eksctl scale nodegroup \
    --cluster qwen3-qlora-cluster \
    --name gpu-nodes \
    --nodes 0 \
    --region us-east-1
```

Scale back up when ready to train:

```bash
eksctl scale nodegroup \
    --cluster qwen3-qlora-cluster \
    --name gpu-nodes \
    --nodes 1 \
    --region us-east-1
```

### Use Spot Instances

For development, modify `cluster-config.yaml`:

```yaml
managedNodeGroups:
  - name: gpu-nodes-spot
    instanceType: g5.12xlarge
    spot: true  # Use spot instances (~60-70% savings)
    desiredCapacity: 1
```

**Warning**: Spot instances can be interrupted! Use checkpoint auto-resume
(`--resume_from_checkpoint=auto`) to recover from interruptions.

### Cost Estimates

| Scenario | Cost |
|----------|------|
| Full training run (~4.5 hours) | `~$26 (GPU) + ~$1 (system) = ~$27` |
| Development (1 hour, then scale to 0) | `~$6` |
| Idle cluster (GPU nodes at 0) | `~$0.20/hr (system node + control plane)` |

## Cleanup

Delete all resources:

```bash
# Delete Kubernetes resources first
kubectl delete namespace ml-training

# Delete the cluster (10-15 minutes)
eksctl delete cluster --name qwen3-qlora-cluster --region us-east-1

# Delete ECR repository
aws ecr delete-repository \
    --repository-name qwen3-qlora-training \
    --region us-east-1 \
    --force
```

## Troubleshooting

### Cluster Creation Fails

```bash
# Check CloudFormation events
aws cloudformation describe-stack-events \
    --stack-name eksctl-qwen3-qlora-cluster-cluster
```

### GPU Nodes Not Ready

```bash
# Check node conditions
kubectl describe node -l node-type=gpu

# Check device plugin logs
kubectl logs -n kube-system -l name=nvidia-device-plugin-ds
```

### Cannot Access Cluster

```bash
# Update kubeconfig
aws eks update-kubeconfig --region us-east-1 --name qwen3-qlora-cluster

# Verify context
kubectl config current-context
```

## Next Steps

After setting up the cluster:

1. Build and push Docker image: `./scripts/build_and_push.sh`
2. Deploy training job: `./scripts/deploy_training.sh`
3. Monitor training: `kubectl logs -f job/qwen3-qlora-training -n ml-training`
