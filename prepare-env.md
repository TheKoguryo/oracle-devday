# Prepare Enviroment

```
# resize boot volume
sudo /usr/libexec/oci-growfs -y

# git
sudo yum install -y git

# docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

sudo yum install -y docker-ce docker-ce-cli containerd.io

sudo systemctl start docker
sudo systemctl enable docker

sudo usermod -aG docker $USER
newgrp docker

docker run hello-world
docker pull container-registry.oracle.com/database/free:latest

# ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull paraphrase-multilingual:latest
ollama pull llama3.1
# CPU Only
ollama run llama3.2
ollama run llama3.2:1b

# python
sudo yum install -y python3.11
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
sudo yum install python3-pip -y
python -m ensurepip
pip install --upgrade pip
```
