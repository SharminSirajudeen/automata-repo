# Complete Guide to Buying and Setting Up vast.ai Instance

## 📋 Prerequisites

Before starting, ensure you have:
- [ ] Credit card for payment
- [ ] Email address for account creation
- [ ] SSH key pair (we'll create one if you don't have it)
- [ ] Basic understanding of terminal/command line

## 🚀 Step-by-Step Guide

### Step 1: Create vast.ai Account

1. **Go to vast.ai**
   ```
   https://vast.ai/
   ```

2. **Click "Sign Up"** (top right corner)
   - Enter your email address
   - Create a strong password
   - Verify your email address

3. **Complete Profile**
   - Add your name
   - Set up 2FA (recommended for security)

### Step 2: Add Credits to Your Account

1. **Navigate to Billing**
   - Click your username (top right)
   - Select "Billing" from dropdown

2. **Add Payment Method**
   - Click "Add Credit"
   - Enter credit card information
   - Add initial credit ($20-50 recommended for testing)

3. **Verify Payment**
   - Credits should appear in your account immediately
   - Check balance in top navigation bar

### Step 3: Choose Your Instance

1. **Go to "Search for Instances"**
   ```
   https://vast.ai/console/create/
   ```

2. **Set Your Requirements**

   For Automata Learning Platform, use these filters:

   **Minimum Requirements:**
   ```
   GPU: Any (not required, but helpful for AI features)
   vCPUs: 4+
   RAM: 8 GB+
   Storage: 50 GB+
   Internet Download: 100+ Mbps
   Internet Upload: 50+ Mbps
   Price: $0.20-0.50/hour (adjust based on budget)
   ```

   **Recommended Specifications:**
   ```
   GPU: RTX 3060 or better (for AI acceleration)
   vCPUs: 8
   RAM: 16 GB
   Storage: 100 GB NVMe
   Internet Download: 500+ Mbps
   Internet Upload: 100+ Mbps
   Docker: Yes
   Direct SSH: Yes
   ```

3. **Apply Filters**
   - ✅ Check "Verified Machines" for reliability
   - ✅ Check "Direct SSH Connection"
   - ✅ Set "Reliability" > 95%
   - ✅ Set your maximum price per hour

### Step 4: Select and Rent Instance

1. **Review Available Instances**
   - Sort by "$/hr" for best value
   - Check the reliability score (aim for >95%)
   - Review the location (closer = lower latency)

2. **Select Your Instance**
   - Click on a suitable instance
   - Review specifications one more time

3. **Configure Instance**
   ```
   Template: PyTorch (includes CUDA if GPU)
   Docker Image: nvidia/cuda:11.8.0-base-ubuntu22.04
   Jupyter: No (we'll use SSH)
   SSH: Yes
   Disk Space: 50 GB minimum
   ```

4. **Set Instance Name**
   ```
   Name: automata-learning-platform
   ```

5. **Click "RENT"**
   - Review the hourly cost
   - Confirm rental

### Step 5: Access Your Instance

1. **Wait for Instance to Start**
   - Status will change from "Starting" to "Running"
   - Usually takes 1-3 minutes

2. **Get Connection Details**
   - Go to "Instances" tab
   - Find your running instance
   - Click "Connect" button

3. **Connection Information**
   You'll see something like:
   ```
   SSH Host: ssh://root@123.45.67.89:22000
   SSH Command: ssh -p 22000 root@123.45.67.89 -L 8080:localhost:8080
   Password: [auto-generated password]
   ```

### Step 6: Connect via SSH

#### Option A: Using Password (Quick Start)

1. **Open Terminal** (or PowerShell on Windows)

2. **Connect to Instance**
   ```bash
   ssh -p 22000 root@123.45.67.89
   ```
   Replace with your actual host and port

3. **Enter Password** when prompted

#### Option B: Using SSH Key (Recommended)

1. **Generate SSH Key** (if you don't have one)
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your-email@example.com"
   ```

2. **Copy Public Key**
   ```bash
   cat ~/.ssh/id_rsa.pub
   ```

3. **Add to vast.ai**
   - Go to Account Settings
   - Click "SSH Keys"
   - Paste your public key
   - Save

4. **Connect with SSH Key**
   ```bash
   ssh -i ~/.ssh/id_rsa -p 22000 root@123.45.67.89
   ```

### Step 7: Initial Server Setup

Once connected, run these commands:

1. **Update System**
   ```bash
   apt update && apt upgrade -y
   ```

2. **Install Essential Tools**
   ```bash
   # Install Docker (if not present)
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   
   # Install Docker Compose
   apt install docker-compose -y
   
   # Install Git
   apt install git -y
   
   # Install Node.js
   curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
   apt install nodejs -y
   
   # Install Python
   apt install python3 python3-pip -y
   
   # Install kubectl
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   chmod +x kubectl
   mv kubectl /usr/local/bin/
   ```

3. **Clone Your Repository**
   ```bash
   git clone https://github.com/your-username/automata-repo.git
   cd automata-repo
   ```

### Step 8: Deploy the Application

1. **Set Up Environment Variables**
   ```bash
   # Copy environment template
   cp .env.example .env.production
   
   # Edit with your values
   nano .env.production
   ```

2. **Run Deployment Script**
   ```bash
   # Make script executable
   chmod +x scripts/deploy-to-vast.sh
   
   # Run deployment
   ./scripts/deploy-to-vast.sh
   ```

3. **Alternative: Manual Docker Deployment**
   ```bash
   # Build and run with Docker Compose
   docker-compose -f docker-compose.prod.yml up -d
   
   # Check status
   docker-compose ps
   ```

### Step 9: Configure Firewall & Networking

1. **Open Required Ports**
   
   In vast.ai console:
   - Click on your instance
   - Go to "Firewall Rules"
   - Add these ports:
   ```
   80 (HTTP)
   443 (HTTPS)
   3000 (Frontend Dev)
   8000 (Backend API)
   5432 (PostgreSQL - only if needed externally)
   6379 (Redis - only if needed externally)
   ```

2. **Test Access**
   ```bash
   # From your local machine
   curl http://YOUR_INSTANCE_IP:8000/health
   ```

### Step 10: Set Up Domain (Optional)

1. **Get Instance IP**
   ```bash
   # On the instance
   curl ifconfig.me
   ```

2. **Configure DNS**
   - Go to your domain provider
   - Add A record pointing to instance IP
   ```
   Type: A
   Name: @ (or subdomain)
   Value: YOUR_INSTANCE_IP
   TTL: 300
   ```

3. **Install SSL Certificate**
   ```bash
   # Install Certbot
   apt install certbot python3-certbot-nginx -y
   
   # Get certificate
   certbot --nginx -d yourdomain.com
   ```

## 💰 Cost Management

### Pricing Breakdown
- **Basic Instance**: $0.10-0.20/hour (~$72-144/month if running 24/7)
- **GPU Instance**: $0.30-0.50/hour (~$216-360/month if running 24/7)
- **Storage**: Usually included, extra at $0.10/GB/month

### Cost-Saving Tips

1. **Stop When Not Using**
   ```bash
   # In vast.ai console
   Click "Stop" on your instance
   # You still pay for storage but not compute
   ```

2. **Use Interruptible Instances**
   - 50-70% cheaper
   - May be stopped with 1-minute notice
   - Good for development/testing

3. **Schedule Automatic Stops**
   ```bash
   # Add cron job to stop at night
   crontab -e
   # Add: 0 22 * * * docker-compose down
   ```

4. **Monitor Usage**
   - Check "Billing" tab regularly
   - Set up spending alerts

## 🔍 Monitoring Your Instance

### Check Instance Status
```bash
# System resources
htop

# Docker containers
docker ps

# Disk usage
df -h

# Network usage
vnstat
```

### View Application Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Performance Monitoring
```bash
# Install monitoring stack
docker run -d \
  --name=netdata \
  -p 19999:19999 \
  -v /etc/passwd:/host/etc/passwd:ro \
  -v /etc/group:/host/etc/group:ro \
  -v /proc:/host/proc:ro \
  -v /sys:/host/sys:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  --cap-add SYS_PTRACE \
  --security-opt apparmor=unconfined \
  netdata/netdata

# Access at http://YOUR_IP:19999
```

## 🚨 Troubleshooting

### Connection Issues
```bash
# If SSH fails, try:
# 1. Check instance is running in vast.ai console
# 2. Verify port number (often not 22)
# 3. Check firewall rules
# 4. Try password instead of key
```

### Docker Issues
```bash
# Reset Docker
systemctl restart docker

# Clean up Docker
docker system prune -a

# Check Docker logs
journalctl -u docker.service
```

### Performance Issues
```bash
# Check CPU/Memory
top

# Check disk I/O
iotop

# Check network
iftop
```

## 📊 Instance Comparison

| Type | vCPUs | RAM | Storage | GPU | Price/hr | Use Case |
|------|-------|-----|---------|-----|----------|----------|
| Basic | 4 | 8GB | 50GB | None | $0.10 | Development |
| Standard | 8 | 16GB | 100GB | None | $0.20 | Production |
| GPU Basic | 8 | 16GB | 100GB | RTX 3060 | $0.30 | AI Features |
| GPU Pro | 16 | 32GB | 200GB | RTX 4090 | $0.80 | Heavy AI |

## ✅ Deployment Checklist

- [ ] Account created and verified
- [ ] Credits added to account
- [ ] Instance selected and rented
- [ ] SSH connection established
- [ ] System packages updated
- [ ] Docker & Docker Compose installed
- [ ] Application cloned from Git
- [ ] Environment variables configured
- [ ] Application deployed and running
- [ ] Firewall rules configured
- [ ] Health checks passing
- [ ] Domain configured (optional)
- [ ] SSL certificate installed (optional)
- [ ] Monitoring set up
- [ ] Backup strategy in place

## 🎯 Next Steps

1. **Test Your Application**
   ```bash
   curl http://YOUR_IP:8000/health
   # Should return {"status": "healthy"}
   ```

2. **Access Frontend**
   - Open browser to `http://YOUR_IP:3000`

3. **Set Up Continuous Deployment**
   - Configure GitHub Actions
   - Set up webhooks for auto-deploy

4. **Configure Backups**
   ```bash
   # Daily database backup
   crontab -e
   # Add: 0 2 * * * docker exec postgres pg_dump -U postgres automata > /backup/db-$(date +\%Y\%m\%d).sql
   ```

## 💡 Pro Tips

1. **Save Money**: Stop instances when not in use
2. **Security**: Change default passwords immediately
3. **Monitoring**: Set up alerts for high CPU/memory usage
4. **Backups**: Regular snapshots of your data
5. **Scaling**: Use Kubernetes for multi-instance deployments

## 📞 Support

- **vast.ai Support**: support@vast.ai
- **Discord Community**: https://discord.gg/vast-ai
- **Documentation**: https://vast.ai/docs

## 🎉 Congratulations!

Your Automata Learning Platform is now running on vast.ai! Access it at:
- Frontend: `http://YOUR_INSTANCE_IP:3000`
- Backend API: `http://YOUR_INSTANCE_IP:8000`
- API Docs: `http://YOUR_INSTANCE_IP:8000/docs`

Remember to monitor your usage and costs regularly!