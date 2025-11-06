# Prepare Environment

```bash
# resize boot volume
sudo /usr/libexec/oci-growfs -y

# docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

sudo yum install -y docker-ce docker-ce-cli containerd.io

sudo systemctl start docker
sudo systemctl enable docker

sudo usermod -aG docker $USER
newgrp docker

docker run hello-world
docker pull container-registry.oracle.com/database/free:23.26.0.0

# ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull paraphrase-multilingual:latest

ollama pull llama3.1

## Laptop - CPU Only
ollama pull llama3.2
ollama pull llama3.2:1b

# python
sudo yum install -y python3.11
sudo yum install python3-pip -y
python3.11 -m ensurepip
pip3.11 install --upgrade pip

cat <<EOF >> ~/.bash_profile
alias pip='pip3.11'
alias python='python3.11'
EOF

source ~/.bash_profile

# git
sudo yum install -y git
```
