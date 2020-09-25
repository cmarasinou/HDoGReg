# Instruction for Docker

# Create image

```bash
docker build -t docker_image_name -f ./Dockerfile.nvidia440 ./
```

# Create container and run

```bash
nvidia-docker  run -it --ipc=host -v ~:/workspace/ docker_image_name
```