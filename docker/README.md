# Instruction for Docker

# Create image

```console
docker build -t docker_image_name -f ./Dockerfile.nvidia440 ./
```

# Create container and run

```console
nvidia-docker  run -it --ipc=host -v ~:/workspace/ docker_image_name
```