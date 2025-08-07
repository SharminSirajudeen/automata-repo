# 🚀 Deploy Your Automata Platform NOW

Choose your deployment option below. All options are ready to go!

## Option 1: vast.ai (GPU + Cheap) - $30-100/month

### Step 1: Create Account (5 minutes)
1. Go to https://vast.ai/
2. Click "Sign Up"
3. Verify email
4. Add $20 credits (minimum)

### Step 2: Rent Instance (3 minutes)
1. Go to https://vast.ai/console/create/
2. Set filters:
   ```
   vCPUs: 4+
   RAM: 8 GB+
   Storage: 50 GB+
   Direct SSH: Yes (checked)
   Max Price: $0.30/hr
   ```
3. Sort by $/hr
4. Click "RENT" on a suitable instance

### Step 3: Deploy (10 minutes)
```bash
# Copy the SSH command from vast.ai console
ssh -p [PORT] root@[IP]

# Once connected, run:
apt update && apt install -y git docker.io docker-compose
git clone https://github.com/[your-username]/automata-repo.git
cd automata-repo

# Quick deploy:
docker-compose up -d

# Your app is now running!
# Frontend: http://[IP]:3000
# Backend: http://[IP]:8000
```

---

## Option 2: Hetzner Cloud (BEST VALUE) - €5-48/month

### Step 1: Create Account (5 minutes)
1. Go to https://www.hetzner.com/cloud
2. Sign up (get €20 free credit!)
3. Verify email

### Step 2: Create Server (2 minutes)
1. Click "New Project" → "Create Server"
2. Choose:
   - Location: Falkenstein (cheapest)
   - Image: Ubuntu 22.04
   - Type: CX21 (2 vCPU, 4GB RAM, €5.82/month)
   - Add SSH key (or use password)
3. Click "Create & Buy Now"

### Step 3: Deploy (5 minutes)
```bash
# SSH to your server
ssh root@[YOUR_IP]

# Install Docker
curl -fsSL https://get.docker.com | sh

# Clone and deploy
git clone https://github.com/[your-username]/automata-repo.git
cd automata-repo
docker-compose up -d

# Done! Access at http://[YOUR_IP]:3000
```

---

## Option 3: RunPod (GPU + Fast) - $50-200/month

### Step 1: Create Account (3 minutes)
1. Go to https://www.runpod.io/
2. Sign up with GitHub/Google
3. Add payment method

### Step 2: Deploy Pod (2 minutes)
1. Click "Deploy Pod"
2. Select GPU: RTX 4090 ($0.34/hr)
3. Choose template: "RunPod Pytorch 2.0"
4. Deploy

### Step 3: Setup (5 minutes)
```bash
# In RunPod terminal:
git clone https://github.com/[your-username]/automata-repo.git
cd automata-repo
docker-compose up -d

# Access via RunPod's proxy URL
```

---

## Option 4: Google Cloud Run (Serverless) - $0-60/month

### Step 1: Setup (10 minutes)
1. Install gcloud: https://cloud.google.com/sdk/docs/install
2. Create project:
```bash
gcloud projects create automata-platform-[random]
gcloud config set project automata-platform-[random]
```

### Step 2: Deploy Backend (3 minutes)
```bash
cd backend
gcloud run deploy automata-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Step 3: Deploy Frontend (3 minutes)
```bash
cd ../frontend
gcloud run deploy automata-frontend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi

# Get URLs from output
```

---

## Option 5: Local Testing First (FREE)

### If you want to test locally before paying:

#### Mac/Linux:
```bash
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop/

# Then run:
cd automata-repo
docker-compose up --build

# Access at http://localhost:3000
```

#### Windows:
```bash
# Install Docker Desktop for Windows
# Download from: https://www.docker.com/products/docker-desktop/

# In PowerShell:
cd automata-repo
docker-compose up --build

# Access at http://localhost:3000
```

---

## 🎯 Quick Decision Guide

| If you want... | Choose... | Cost | Setup Time |
|---------------|-----------|------|------------|
| Cheapest option | Hetzner | €5.82/mo | 10 min |
| GPU for AI | vast.ai | $30-100/mo | 15 min |
| Most reliable | Google Cloud Run | $0-60/mo | 15 min |
| Fastest setup | RunPod | $50-200/mo | 10 min |
| Test first | Local Docker | FREE | 5 min |

## 🔥 Fastest Option: Hetzner (€20 free credit!)

```bash
# Complete deployment in 5 commands:
# 1. Create Hetzner account (get €20 free)
# 2. Create CX21 server
# 3. SSH to server and run:

curl -fsSL https://get.docker.com | sh
git clone [your-repo]
cd automata-repo
docker-compose up -d

# Done! Your platform is live!
```

## Need Help?

- vast.ai issues? Check: https://vast.ai/docs
- Hetzner issues? Check: https://docs.hetzner.com
- Docker issues? Run: `docker-compose logs`
- Application issues? Check: `docker ps` and `docker logs [container]`

## 🎉 Your Platform is Ready to Deploy!

Pick any option above and your platform will be live in 15 minutes or less!