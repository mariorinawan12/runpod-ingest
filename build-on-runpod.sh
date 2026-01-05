#!/bin/bash
# ============================================================================
# Build Docker Image on RunPod CPU Pod
# ============================================================================
# This script automates building and pushing your Docker image from RunPod.
# 
# Usage:
#   1. SSH into your RunPod CPU pod
#   2. Run: bash build-on-runpod.sh
#
# Prerequisites:
#   - Docker Hub account
#   - GitHub repo with your code
# ============================================================================

set -e  # Exit on error

echo "🚀 RunPod Docker Build Script"
echo "=============================="
echo ""

# ============================================================================
# CONFIGURATION - EDIT THESE!
# ============================================================================
GITHUB_REPO="https://github.com/mariorinawan12/runpod-ingest.git"
DOCKERHUB_USERNAME="mariorinawan12"
IMAGE_NAME="runpod-ingest"
BRANCH="main"  # or "master" depending on your repo

# ============================================================================
# STEP 1: Install Docker (if not installed)
# ============================================================================
echo "📦 Step 1: Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    echo "✅ Docker installed!"
else
    echo "✅ Docker already installed!"
fi

# ============================================================================
# STEP 2: Login to Docker Hub
# ============================================================================
echo ""
echo "🔐 Step 2: Login to Docker Hub"
echo "Enter your Docker Hub credentials:"
docker login

# ============================================================================
# STEP 3: Clone/Pull GitHub Repo
# ============================================================================
echo ""
echo "📥 Step 3: Getting code from GitHub..."
if [ -d "runpod-ingest" ]; then
    echo "Repo exists. Pulling latest changes..."
    cd runpod-ingest
    git pull origin $BRANCH
else
    echo "Cloning repo..."
    git clone $GITHUB_REPO
    cd runpod-ingest
fi

# ============================================================================
# STEP 4: Build Docker Image
# ============================================================================
echo ""
echo "🔨 Step 4: Building Docker image..."
echo "This will take 10-15 minutes (downloading PyTorch + CUDA base image)..."
echo ""

# Get version from git tag or use 'latest'
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "latest")
echo "Building version: $VERSION"

docker build -t $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION .

# Also tag as latest
docker tag $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION $DOCKERHUB_USERNAME/$IMAGE_NAME:latest

echo ""
echo "✅ Build complete!"

# ============================================================================
# STEP 5: Push to Docker Hub
# ============================================================================
echo ""
echo "📤 Step 5: Pushing to Docker Hub..."
docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION
docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:latest

echo ""
echo "✅ Push complete!"

# ============================================================================
# STEP 6: Cleanup (optional)
# ============================================================================
echo ""
echo "🧹 Step 6: Cleanup old images (optional)..."
read -p "Remove old Docker images to free space? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker system prune -af
    echo "✅ Cleanup complete!"
else
    echo "⏭️  Skipped cleanup"
fi

# ============================================================================
# DONE!
# ============================================================================
echo ""
echo "🎉 SUCCESS!"
echo "=========================================="
echo "Image pushed to Docker Hub:"
echo "  - $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION"
echo "  - $DOCKERHUB_USERNAME/$IMAGE_NAME:latest"
echo ""
echo "Next steps:"
echo "1. Go to RunPod Serverless console"
echo "2. Create/Update endpoint with image:"
echo "   $DOCKERHUB_USERNAME/$IMAGE_NAME:latest"
echo "3. Test your endpoint!"
echo "=========================================="
