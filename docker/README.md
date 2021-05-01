# Instructions for Docker 

## Create Image

```bash
docker build -t docker_image_name -f ./Dockerfile.nvidia440 ./
```

## Create and Run Container

```bash
docker run -it --ipc=host --gpus all -v ~:/workspace/ docker_image_name
```