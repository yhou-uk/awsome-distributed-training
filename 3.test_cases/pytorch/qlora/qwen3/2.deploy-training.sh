#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Deploy the QLoRA training Job to EKS.
#
# Usage:
#   export IMAGE=<account>.dkr.ecr.<region>.amazonaws.com/qwen3-qlora-training:latest
#   ./2.deploy-training.sh
#
# Prerequisites:
#   - EKS cluster running with GPU nodes (see 0.setup-cluster.sh)
#   - NVIDIA device plugin installed
#   - Docker image built and pushed (see 1.build-image.sh)

set -euo pipefail

###########################
###### User Variables #####
###########################

export NAMESPACE=${NAMESPACE:-ml-training}
export NUM_GPUS=${NUM_GPUS:-4}

# IMAGE must be set
if [ -z "${IMAGE:-}" ]; then
    echo "ERROR: IMAGE environment variable not set."
    echo "Run: export IMAGE=<account>.dkr.ecr.<region>.amazonaws.com/qwen3-qlora-training:latest"
    exit 1
fi

###########################
###### Create Namespace ###
###########################

echo "Ensuring namespace ${NAMESPACE} exists..."
kubectl create namespace ${NAMESPACE} 2>/dev/null || true

###########################
###### Deploy Job #########
###########################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Deploying training job..."
echo "  Image:     ${IMAGE}"
echo "  GPUs:      ${NUM_GPUS}"
echo "  Namespace: ${NAMESPACE}"
echo ""

envsubst < "${SCRIPT_DIR}/kubernetes/qwen3_8b-qlora.yaml" | kubectl apply -f -

echo ""
echo "Training job deployed. Monitor with:"
echo "  kubectl get pods -n ${NAMESPACE} -w"
echo "  kubectl logs -f job/qwen3-qlora-training -n ${NAMESPACE}"
