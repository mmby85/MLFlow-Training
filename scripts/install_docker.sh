#!/bin/bash

# Update package lists
sudo apt-get update

# Install required packages to allow apt to use repositories over HTTPS
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up the stable Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package lists again
sudo apt-get update

# Install Docker Engine
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Verify Docker installation
sudo docker run hello-world

echo "Docker installed successfully!"

# Install Docker Compose
sudo apt-get update
sudo apt-get install -y docker-compose-plugin

# Verify Docker Compose installation
docker compose version

echo "Docker Compose installed successfully!"

echo "Installation complete!"