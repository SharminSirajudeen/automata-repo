#!/bin/bash
set -euo pipefail

# Kubernetes Cluster Setup Script for vast.ai
# Configures a production-ready K8s cluster on GPU instances

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
K8S_VERSION="${K8S_VERSION:-1.28}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-containerd}"
CNI_PLUGIN="${CNI_PLUGIN:-cilium}"
NVIDIA_OPERATOR_VERSION="${NVIDIA_OPERATOR_VERSION:-v23.9.1}"
CERT_MANAGER_VERSION="${CERT_MANAGER_VERSION:-v1.13.2}"
INGRESS_NGINX_VERSION="${INGRESS_NGINX_VERSION:-4.8.3}"

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
    fi
}

# Detect vast.ai environment
detect_vast_ai() {
    log "Detecting vast.ai environment..."
    
    if [[ -n "${VAST_AI_INSTANCE_ID:-}" ]]; then
        log "Running on vast.ai instance: ${VAST_AI_INSTANCE_ID}"
        export IS_VAST_AI=true
    else
        log "Not running on vast.ai, assuming generic environment"
        export IS_VAST_AI=false
    fi
    
    # Detect GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        log "Detected ${GPU_COUNT} GPU(s) with ${GPU_MEMORY}MB memory each"
        export HAS_GPU=true
        export GPU_COUNT
        export GPU_MEMORY
    else
        warning "No NVIDIA GPUs detected"
        export HAS_GPU=false
    fi
}

# Update system packages
update_system() {
    log "Updating system packages..."
    
    sudo apt-get update
    sudo apt-get upgrade -y
    
    # Install essential packages
    sudo apt-get install -y \
        curl \
        wget \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        software-properties-common \
        unzip \
        git \
        htop \
        vim \
        jq \
        net-tools
    
    success "System packages updated"
}

# Install Docker
install_docker() {
    log "Installing Docker..."
    
    if command -v docker &> /dev/null; then
        warning "Docker already installed, skipping..."
        return
    fi
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    # Configure Docker for production
    sudo mkdir -p /etc/docker
    cat <<EOF | sudo tee /etc/docker/daemon.json
{
    "exec-opts": ["native.cgroupdriver=systemd"],
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "storage-opts": [
        "overlay2.override_kernel_check=true"
    ],
    "insecure-registries": ["registry.vast.ai"],
    "registry-mirrors": ["https://registry.vast.ai"]
}
EOF
    
    sudo systemctl enable docker
    sudo systemctl restart docker
    
    success "Docker installed"
}

# Install NVIDIA Container Runtime (if GPU available)
install_nvidia_runtime() {
    if [[ "${HAS_GPU}" != "true" ]]; then
        log "No GPU detected, skipping NVIDIA runtime installation"
        return
    fi
    
    log "Installing NVIDIA Container Runtime..."
    
    # Add NVIDIA repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime
    
    # Configure Docker for NVIDIA
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    success "NVIDIA Container Runtime installed"
}

# Install Kubernetes tools
install_k8s_tools() {
    log "Installing Kubernetes tools..."
    
    # Add Kubernetes repository
    curl -fsSL https://pkgs.k8s.io/core:/stable:/v${K8S_VERSION}/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
    echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v${K8S_VERSION}/deb/ /" | sudo tee /etc/apt/sources.list.d/kubernetes.list
    
    sudo apt-get update
    sudo apt-get install -y kubelet kubeadm kubectl
    sudo apt-mark hold kubelet kubeadm kubectl
    
    success "Kubernetes tools installed"
}

# Initialize Kubernetes cluster
init_k8s_cluster() {
    log "Initializing Kubernetes cluster..."
    
    # Check if already initialized
    if kubectl cluster-info &> /dev/null; then
        warning "Kubernetes cluster already initialized"
        return
    fi
    
    # Get the primary IP address
    PRIMARY_IP=$(ip route get 8.8.8.8 | awk '{print $7; exit}')
    
    # Initialize cluster
    sudo kubeadm init \
        --apiserver-advertise-address=${PRIMARY_IP} \
        --pod-network-cidr=10.244.0.0/16 \
        --service-cidr=10.96.0.0/12 \
        --kubernetes-version=v${K8S_VERSION}.0
    
    # Set up kubeconfig for regular user
    mkdir -p $HOME/.kube
    sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
    sudo chown $(id -u):$(id -g) $HOME/.kube/config
    
    # Remove master taint for single-node setup
    kubectl taint nodes --all node-role.kubernetes.io/control-plane- || true
    
    success "Kubernetes cluster initialized"
}

# Install CNI plugin
install_cni() {
    log "Installing CNI plugin: ${CNI_PLUGIN}..."
    
    case ${CNI_PLUGIN} in
        "cilium")
            # Install Cilium CLI
            CILIUM_CLI_VERSION=$(curl -s https://raw.githubusercontent.com/cilium/cilium-cli/main/stable.txt)
            CLI_ARCH=amd64
            if [ "$(uname -m)" = "aarch64" ]; then CLI_ARCH=arm64; fi
            curl -L --fail --remote-name-all https://github.com/cilium/cilium-cli/releases/download/${CILIUM_CLI_VERSION}/cilium-linux-${CLI_ARCH}.tar.gz{,.sha256sum}
            sha256sum --check cilium-linux-${CLI_ARCH}.tar.gz.sha256sum
            sudo tar xzvfC cilium-linux-${CLI_ARCH}.tar.gz /usr/local/bin
            rm cilium-linux-${CLI_ARCH}.tar.gz{,.sha256sum}
            
            # Install Cilium
            cilium install \
                --set operator.replicas=1 \
                --set tunnel=vxlan \
                --set ipam.mode=kubernetes \
                --set kubeProxyReplacement=strict
            
            # Wait for Cilium to be ready
            cilium status --wait
            ;;
        "flannel")
            kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
            ;;
        "calico")
            kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.4/manifests/tigera-operator.yaml
            kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.4/manifests/custom-resources.yaml
            ;;
        *)
            error "Unsupported CNI plugin: ${CNI_PLUGIN}"
            ;;
    esac
    
    success "CNI plugin ${CNI_PLUGIN} installed"
}

# Install NVIDIA GPU Operator (if GPU available)
install_gpu_operator() {
    if [[ "${HAS_GPU}" != "true" ]]; then
        log "No GPU detected, skipping GPU Operator installation"
        return
    fi
    
    log "Installing NVIDIA GPU Operator..."
    
    # Add NVIDIA Helm repository
    helm repo add nvidia https://nvidia.github.io/gpu-operator
    helm repo update
    
    # Create namespace
    kubectl create namespace gpu-operator-resources --dry-run=client -o yaml | kubectl apply -f -
    
    # Install GPU Operator
    helm install gpu-operator nvidia/gpu-operator \
        --namespace gpu-operator-resources \
        --version ${NVIDIA_OPERATOR_VERSION} \
        --set operator.defaultRuntime=containerd \
        --set toolkit.env[0].name=CONTAINERD_CONFIG \
        --set toolkit.env[0].value=/etc/containerd/config.toml \
        --set toolkit.env[1].name=CONTAINERD_SOCKET \
        --set toolkit.env[1].value=/run/containerd/containerd.sock \
        --set toolkit.env[2].name=CONTAINERD_RUNTIME_CLASS \
        --set toolkit.env[2].value=nvidia \
        --set toolkit.env[3].name=CONTAINERD_SET_AS_DEFAULT \
        --set-string toolkit.env[3].value=true
    
    # Wait for GPU Operator to be ready
    log "Waiting for GPU Operator to be ready..."
    kubectl wait --for=condition=ready pod -l app=nvidia-operator-validator -n gpu-operator-resources --timeout=600s
    
    success "NVIDIA GPU Operator installed"
}

# Install Helm
install_helm() {
    log "Installing Helm..."
    
    if command -v helm &> /dev/null; then
        warning "Helm already installed, skipping..."
        return
    fi
    
    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    
    success "Helm installed"
}

# Install cert-manager
install_cert_manager() {
    log "Installing cert-manager..."
    
    # Add cert-manager repository
    helm repo add jetstack https://charts.jetstack.io
    helm repo update
    
    # Create namespace
    kubectl create namespace cert-manager --dry-run=client -o yaml | kubectl apply -f -
    
    # Install cert-manager
    helm install cert-manager jetstack/cert-manager \
        --namespace cert-manager \
        --version ${CERT_MANAGER_VERSION} \
        --set installCRDs=true \
        --set global.leaderElection.namespace=cert-manager
    
    # Wait for cert-manager to be ready
    kubectl wait --for=condition=Available deployment --all -n cert-manager --timeout=300s
    
    # Create ClusterIssuer for Let's Encrypt
    cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@${DOMAIN:-automata.vast.ai}
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
    
    success "cert-manager installed"
}

# Install NGINX Ingress Controller
install_nginx_ingress() {
    log "Installing NGINX Ingress Controller..."
    
    # Add ingress-nginx repository
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    helm repo update
    
    # Install NGINX Ingress Controller
    helm install ingress-nginx ingress-nginx/ingress-nginx \
        --namespace ingress-nginx \
        --create-namespace \
        --version ${INGRESS_NGINX_VERSION} \
        --set controller.service.type=LoadBalancer \
        --set controller.service.externalTrafficPolicy=Local \
        --set controller.config.use-proxy-protocol=false \
        --set controller.config.compute-full-forwarded-for=true \
        --set controller.config.use-forwarded-headers=true \
        --set controller.metrics.enabled=true \
        --set controller.metrics.serviceMonitor.enabled=true \
        --set controller.podSecurityContext.runAsUser=101 \
        --set controller.podSecurityContext.runAsGroup=82 \
        --set controller.podSecurityContext.fsGroup=82
    
    # Wait for ingress controller to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=ingress-nginx -n ingress-nginx --timeout=300s
    
    success "NGINX Ingress Controller installed"
}

# Install metrics server
install_metrics_server() {
    log "Installing Metrics Server..."
    
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
    
    # Patch metrics server for vast.ai (disable TLS verification)
    kubectl patch deployment metrics-server -n kube-system --type='merge' -p='{
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": "metrics-server",
                        "args": [
                            "--cert-dir=/tmp",
                            "--secure-port=4443",
                            "--kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname",
                            "--kubelet-use-node-status-port",
                            "--metric-resolution=15s",
                            "--kubelet-insecure-tls"
                        ]
                    }]
                }
            }
        }
    }'
    
    # Wait for metrics server to be ready
    kubectl wait --for=condition=Available deployment metrics-server -n kube-system --timeout=300s
    
    success "Metrics Server installed"
}

# Configure storage
configure_storage() {
    log "Configuring storage..."
    
    # Create local storage class for vast.ai
    cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Retain
parameters:
  type: gp3
  fsType: ext4
EOF
    
    success "Storage configured"
}

# Verify cluster setup
verify_cluster() {
    log "Verifying cluster setup..."
    
    # Check cluster info
    kubectl cluster-info
    
    # Check nodes
    kubectl get nodes -o wide
    
    # Check system pods
    kubectl get pods -A
    
    # Check storage classes
    kubectl get storageclass
    
    # Test GPU if available
    if [[ "${HAS_GPU}" == "true" ]]; then
        log "Testing GPU access..."
        cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: Never
  containers:
  - name: gpu-test
    image: nvidia/cuda:11.8-base-ubuntu20.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF
        
        # Wait for pod to complete
        kubectl wait --for=condition=Completed pod/gpu-test --timeout=120s
        kubectl logs gpu-test
        kubectl delete pod gpu-test
    fi
    
    success "Cluster verification complete"
}

# Setup firewall rules for vast.ai
setup_firewall() {
    log "Setting up firewall rules..."
    
    # Install ufw if not present
    sudo apt-get install -y ufw
    
    # Reset firewall rules
    sudo ufw --force reset
    
    # Allow SSH
    sudo ufw allow ssh
    
    # Allow Kubernetes API server
    sudo ufw allow 6443/tcp
    
    # Allow kubelet API
    sudo ufw allow 10250/tcp
    
    # Allow kube-scheduler
    sudo ufw allow 10259/tcp
    
    # Allow kube-controller-manager
    sudo ufw allow 10257/tcp
    
    # Allow etcd
    sudo ufw allow 2379:2380/tcp
    
    # Allow NodePort services
    sudo ufw allow 30000:32767/tcp
    
    # Allow HTTP/HTTPS
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    
    # Allow pod network
    sudo ufw allow from 10.244.0.0/16
    sudo ufw allow from 10.96.0.0/12
    
    # Enable firewall
    sudo ufw --force enable
    
    success "Firewall configured"
}

# Main setup function
main() {
    log "Starting Kubernetes cluster setup for vast.ai"
    
    check_root
    detect_vast_ai
    update_system
    install_docker
    install_nvidia_runtime
    install_k8s_tools
    init_k8s_cluster
    install_helm
    install_cni
    
    if [[ "${HAS_GPU}" == "true" ]]; then
        install_gpu_operator
    fi
    
    install_cert_manager
    install_nginx_ingress
    install_metrics_server
    configure_storage
    
    if [[ "${IS_VAST_AI}" == "true" ]]; then
        setup_firewall
    fi
    
    verify_cluster
    
    success "Kubernetes cluster setup complete!"
    success "Cluster is ready for Automata Platform deployment"
    
    log "Next steps:"
    log "1. Configure your secrets: export POSTGRES_PASSWORD=... SECRET_KEY=... etc."
    log "2. Run the deployment script: ./scripts/deploy-vast.sh"
    log "3. Access your cluster: kubectl get nodes"
    
    if [[ "${HAS_GPU}" == "true" ]]; then
        log "GPU support is enabled - your cluster can run AI workloads!"
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi