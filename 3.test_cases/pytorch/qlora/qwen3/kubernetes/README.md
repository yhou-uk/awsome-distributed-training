# Kubernetes (EKS) Deployment

Instructions for deploying QLoRA training on Amazon EKS.

## Prerequisites

1. **EKS cluster** with a GPU node group (g5.12xlarge recommended)

   Create using the provided cluster config:
   ```bash
   # Edit kubernetes/eks-cluster.yaml and replace PLACEHOLDER_* values:
   #   PLACEHOLDER_AWS_REGION  -> your region (e.g., us-east-1)
   #   PLACEHOLDER_AZ_1        -> first AZ (e.g., us-east-1a)
   #   PLACEHOLDER_AZ_2        -> second AZ with g5 capacity (e.g., us-east-1f)
   eksctl create cluster -f kubernetes/eks-cluster.yaml
   ```

2. **NVIDIA device plugin**
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml
   ```

3. **Kubeflow Training Operator**
   ```bash
   kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
   ```

4. **Namespace and storage**
   ```bash
   kubectl create namespace ml-training

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

   kubectl apply -f - <<EOF
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: training-storage
     namespace: ml-training
   spec:
     accessModes:
       - ReadWriteOnce
     storageClassName: gp3-storage
     resources:
       requests:
         storage: 200Gi
   EOF
   ```

## Build and Push Docker Image

```bash
# Set variables
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1

# Create ECR repository (first time only)
aws ecr create-repository \
    --repository-name qwen3-qlora-training \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true

# Authenticate to ECR
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build and push (from the qwen3/ directory)
docker build -t ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/qwen3-qlora-training:latest .
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/qwen3-qlora-training:latest
```

Or use the provided script:
```bash
./0.build-image.sh
```

## Deploy Training Job

```bash
export IMAGE=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/qwen3-qlora-training:latest
export NUM_GPUS=4
export NAMESPACE=ml-training

envsubst < kubernetes/qwen3_8b-qlora.yaml | kubectl apply -f -
```

Or use the provided script:
```bash
./1.deploy-training.sh
```

## Monitor Training

```bash
# Watch pod status
kubectl get pods -n ml-training -w

# Stream logs
kubectl logs -f -n ml-training qwen3-qlora-training-master-0

# Check GPU utilization (exec into the pod)
kubectl exec -it -n ml-training qwen3-qlora-training-master-0 -- nvidia-smi
```

## Cleanup

```bash
kubectl delete pytorchjob qwen3-qlora-training -n ml-training
```

To delete the entire cluster:
```bash
kubectl delete namespace ml-training
eksctl delete cluster --name qwen3-qlora-cluster --region $AWS_REGION
```

Or use the provided script:
```bash
./2.cleanup.sh
```

## Cost Optimization

Scale GPU nodes to zero when not training:
```bash
eksctl scale nodegroup \
    --cluster qwen3-qlora-cluster \
    --name gpu-g5 \
    --nodes 0 \
    --region $AWS_REGION
```
