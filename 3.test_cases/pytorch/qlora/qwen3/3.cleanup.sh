#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Cleanup training resources from EKS.
#
# Usage:
#   ./2.cleanup.sh

set -euo pipefail

###########################
###### User Variables #####
###########################

NAMESPACE=${NAMESPACE:-ml-training}

###########################
###### Delete Job #########
###########################

echo "Deleting PyTorchJob..."
kubectl delete pytorchjob qwen3-qlora-training -n ${NAMESPACE} 2>/dev/null || true

echo ""
echo "Training job deleted."
echo ""
echo "To delete the namespace and all resources:"
echo "  kubectl delete namespace ${NAMESPACE}"
echo ""
echo "To delete the EKS cluster entirely:"
echo "  eksctl delete cluster --name qwen3-qlora-cluster --region \$AWS_REGION"
echo ""
echo "To delete the ECR repository:"
echo "  aws ecr delete-repository --repository-name qwen3-qlora-training --region \$AWS_REGION --force"
