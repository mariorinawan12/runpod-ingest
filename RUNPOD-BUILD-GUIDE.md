# 🚀 Build Docker Image on RunPod - Step by Step Guide

Karena GitLab shared runner ga cukup disk space (14GB limit), kita build di RunPod CPU pod.

## 💰 Cost Estimate

- **RunPod CPU Pod**: ~$0.10-0.20 per build (15-20 minutes)
- **One-time setup**: ~5 minutes
- **Future builds**: Just SSH and run script!

---

## 📋 Prerequisites

1. ✅ RunPod account (https://runpod.io)
2. ✅ Docker Hub account (https://hub.docker.com)
3. ✅ GitHub repo with your code (https://github.com/mariorinawan12/runpod-ingest)
4. ✅ Docker Hub username: **mariorinawan** (not mariorinawan12!)

---

## 🎯 Step-by-Step Instructions

### STEP 1: Deploy RunPod CPU Pod

1. Go to https://www.runpod.io/console/pods
2. Click **"+ Deploy"**
3. Select **"CPU"** tab (cheaper for building)
4. Choose template:
   - **Template**: `RunPod Pytorch` or `Ubuntu + Docker`
   - **vCPU**: 4-8 cores (faster build)
   - **RAM**: 16GB minimum
   - **Disk**: 50GB minimum (for Docker layers)
5. Click **"Deploy On-Demand"**
6. Wait ~30 seconds for pod to start

**Cost**: ~$0.20-0.40 per hour (you'll only use 15-20 minutes)

---

### STEP 2: Connect to Pod via SSH

1. In RunPod console, find your pod
2. Click **"Connect"** → **"Start SSH Terminal"**
3. Copy the SSH command (looks like):
   ```bash
   ssh root@<pod-id>.runpod.io -p 12345 -i ~/.ssh/id_ed25519
   ```
4. Run it in your terminal (or use RunPod's web terminal)

You should see:
```
root@<pod-id>:~#
```

---

### STEP 3: Download Build Script

In the SSH terminal, run:

```bash
# Download the build script
curl -o build-on-runpod.sh https://raw.githubusercontent.com/mariorinawan12/runpod-ingest/main/build-on-runpod.sh

# Make it executable
chmod +x build-on-runpod.sh
```

**OR** if you prefer manual:

```bash
# Clone your repo
git clone https://github.com/mariorinawan12/runpod-ingest.git
cd runpod-ingest

# Use the script from repo
chmod +x build-on-runpod.sh
```

---

### STEP 4: Run Build Script

```bash
bash build-on-runpod.sh
```

The script will:
1. ✅ Check/install Docker
2. ✅ Ask for Docker Hub login (enter username + password/token)
3. ✅ Clone/pull your GitHub repo
4. ✅ Build Docker image (~15 minutes)
5. ✅ Push to Docker Hub
6. ✅ Cleanup (optional)

**What you'll see:**

```
🚀 RunPod Docker Build Script
==============================

📦 Step 1: Checking Docker installation...
✅ Docker already installed!

🔐 Step 2: Login to Docker Hub
Enter your Docker Hub credentials:
Username: mariorinawan
Password: [enter your Docker Hub token]
✅ Login Succeeded!

📥 Step 3: Getting code from GitHub...
Cloning repo...
✅ Code downloaded!

🔨 Step 4: Building Docker image...
This will take 10-15 minutes...
[... lots of Docker output ...]
✅ Build complete!

📤 Step 5: Pushing to Docker Hub...
✅ Push complete!

🎉 SUCCESS!
Image pushed: mariorinawan/runpod-ingest:latest
```

---

### STEP 5: Verify on Docker Hub

1. Go to https://hub.docker.com/r/mariorinawan/runpod-ingest
2. Check that **"latest"** tag exists
3. Check the size (~13GB)

---

### STEP 6: Stop RunPod Pod (Save Money!)

**IMPORTANT**: Stop the pod to avoid charges!

1. Go back to RunPod console
2. Find your pod
3. Click **"Stop"** or **"Terminate"**

**Cost saved**: Stopping immediately after build = ~$0.10-0.20 total

---

## 🔄 Future Builds (After Code Changes)

When you update your code and want to rebuild:

### Option A: Quick Rebuild (Recommended)

```bash
# 1. Start the same pod again (or deploy new one)
# 2. SSH into pod
# 3. Run:
cd runpod-ingest
git pull origin main
docker build -t mariorinawan/runpod-ingest:latest .
docker push mariorinawan/runpod-ingest:latest
```

### Option B: Use Script Again

```bash
bash build-on-runpod.sh
```

---

## 🎯 Workflow Summary

```
┌─────────────────┐
│ Edit Code       │
│ (Local/Laptop)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Commit & Push   │
│ to GitHub       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SSH to RunPod   │
│ Run build script│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Image on        │
│ Docker Hub      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Deploy to       │
│ RunPod Serverless│
└─────────────────┘
```

---

## 🐛 Troubleshooting

### Error: "docker: command not found"

```bash
# Install Docker manually
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

### Error: "no space left on device"

```bash
# Clean up Docker
docker system prune -af

# Or choose bigger disk when deploying pod (50GB+)
```

### Error: "permission denied"

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Error: "git: command not found"

```bash
# Install git
apt-get update && apt-get install -y git
```

---

## 💡 Tips

1. **Save SSH command**: Bookmark the SSH command for quick access
2. **Use same pod**: Keep the pod running if you're doing multiple builds
3. **Automate**: Create a GitHub webhook to trigger builds (advanced)
4. **Monitor costs**: Check RunPod billing dashboard regularly

---

## 🎉 Done!

Your Docker image is now on Docker Hub and ready to deploy to RunPod Serverless!

Next: Go to RunPod Serverless console and create/update your endpoint with:
```
mariorinawan/runpod-ingest:latest
```

---

## 📞 Need Help?

- RunPod Discord: https://discord.gg/runpod
- Docker Hub: https://hub.docker.com/r/mariorinawan/runpod-ingest
- GitHub Issues: https://github.com/mariorinawan12/runpod-ingest/issues
