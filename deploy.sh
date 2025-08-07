#!/bin/bash

# Stop and remove existing container if it exists
docker stop iris-api-container 2>/dev/null
docker rm iris-api-container 2>/dev/null

# Pull the latest image from Docker Hub
docker pull susmita161/iris-api:latest

# Run the container
docker run -d -p 8000:8000 --name iris-api-container susmita161/iris-api:latest

echo "✅ iris-api is running at http://localhost:8000"
