#!/bin/bash
# ============================================================
# EKS Cluster Setup Script for QLoRA Training
# ============================================================
# This script automates the creation and configuration of an
# Amazon EKS cluster for QLoRA fine-tuning of Qwen3-8B.
#
# Prerequisites:
#   - AWS CLI configured with appropriate IAM permissions
#   - eksctl installed
#   - kubectl installed
#
# Usage:
#   chmod +x setup-cluster.sh
#   ./setup-cluster.sh
#
# ============================================================

set -e  # Exit on any error

# Configuration
CLUSTER_NAME="qwen3-qlora-cluster"
REGION="us-east-1"
ECR_REPO_NAME="qwen3-qlora-training"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================
# Step 1: Verify Prerequisites
# ============================================================
echo ""
echo "========================================"
echo "QLoRA Training EKS Cluster Setup"
echo "========================================"
echo ""

log_info "Checking prerequisites..."

# Check eksctl
if ! command -v eksctl &> /dev/null; then
    log_error "eksctl not found. Please install from: https://eksctl.io/installation/"
    exit 1
fi
log_info "eksctl version: $(eksctl version)"

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl not found. Please install from: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi
log_info "kubectl version: $(kubectl version --client --short 2>/dev/null || kubectl version --client)"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    log_error "AWS CLI not found. Please install from: https://aws.amazon.com/cli/"
    exit 1
fi
log_info "AWS CLI version: $(aws --version)"

# ============================================================
# Step 2: Verify AWS Credentials
# ============================================================
log_info "Verifying AWS credentials..."

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
if [ -z "$AWS_ACCOUNT_ID" ]; then
    log_error "Failed to get AWS account ID. Please configure AWS credentials."
    exit 1
fi
log_info "Using AWS Account: $AWS_ACCOUNT_ID"

AWS_REGION=$(aws configure get region)
if [ -z "$AWS_REGION" ]; then
    log_warn "No default region configured. Using: $REGION"
else
    log_info "AWS Region: $AWS_REGION"
    REGION=$AWS_REGION
fi

# ============================================================
# Step 3: Create EKS Cluster
# ============================================================
echo ""
log_info "Creating EKS cluster: $CLUSTER_NAME"
log_info "This will take 15-20 minutes. Please be patient..."
echo ""

eksctl create cluster -f "$SCRIPT_DIR/kubernetes/eks-cluster.yaml"

# Update kubeconfig
log_info "Updating kubeconfig..."
aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME

# ============================================================
# Step 4: Install NVIDIA Device Plugin
# ============================================================
echo ""
log_info "Installing NVIDIA device plugin for Kubernetes..."

# Apply the NVIDIA device plugin DaemonSet
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml

# Wait for the device plugin to be ready
log_info "Waiting for NVIDIA device plugin to be ready..."
kubectl rollout status daemonset/nvidia-device-plugin-daemonset -n kube-system --timeout=300s

# ============================================================
# Step 5: Verify GPU Nodes
# ============================================================
echo ""
log_info "Verifying GPU availability..."

# Wait a bit for GPU nodes to be fully ready
sleep 30

# Check GPU nodes
GPU_NODES=$(kubectl get nodes -l nvidia.com/gpu=true -o jsonpath='{.items[*].metadata.name}')
if [ -z "$GPU_NODES" ]; then
    log_warn "No GPU nodes found yet. They may still be initializing."
else
    log_info "GPU nodes available: $GPU_NODES"
fi

# Check GPU resources
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu

# ============================================================
# Step 6: Create ECR Repository
# ============================================================
echo ""
log_info "Creating ECR repository: $ECR_REPO_NAME..."

aws ecr create-repository \
    --repository-name $ECR_REPO_NAME \
    --region $REGION \
    --image-scanning-configuration scanOnPush=true \
    2>/dev/null || log_warn "ECR repository may already exist"

ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO_NAME"
log_info "ECR repository URI: $ECR_URI"

# ============================================================
# Step 7: Create Storage Class
# ============================================================
echo ""
log_info "Creating gp3 storage class..."

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
allowVolumeExpansion: true
EOF

# ============================================================
# Step 8: Create ML Training Namespace
# ============================================================
echo ""
log_info "Creating ml-training namespace..."

kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ml-training
  labels:
    project: qwen3-qlora
EOF

# ============================================================
# Step 9: Create Persistent Volume Claim
# ============================================================
echo ""
log_info "Creating PersistentVolumeClaim for training storage..."

kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-storage
  namespace: ml-training
  labels:
    project: qwen3-qlora
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: gp3-storage
  resources:
    requests:
      storage: 100Gi
EOF

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================"
echo "EKS Cluster Setup Complete!"
echo "========================================"
echo ""
log_info "Cluster name: $CLUSTER_NAME"
log_info "Region: $REGION"
log_info "ECR Repository: $ECR_URI"
echo ""
log_info "Next steps:"
echo "  1. Build and push Docker image: ./1.build-image.sh"
echo "  2. Deploy training job: ./2.deploy-training.sh"
echo "  3. Monitor training: kubectl logs -f job/qwen3-qlora-training -n ml-training"
echo ""
log_info "To delete the cluster when done:"
echo "  eksctl delete cluster --name=$CLUSTER_NAME --region=$REGION"
echo ""
