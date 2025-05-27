# To build a docker image
sudo docker build -t <image_name> -f docker/Dockerfile .


# To pull image from docker hub
sudo docker pull <image_name>


# Just basic (detached) run (if you are satisfied with default training config.py)
sudo docker run --gpus all -d \
  -v <volume_name>:/<folder_name> \
  -e HUGGING_FACE_HUB_TOKEN=<hf_token> \
  <image_name>


# Example of running (detached) container with different variables in training config.py
sudo docker run --gpus all -d \
  -v <volume_name>:/<folder_name> \
  -e LOAD_FROM_HF="True" \
  -e CHECKPOINT_NUM="3000" \
  -e RESUME_FROM_CHECKPOINT="True" \
  -e BATCH_SIZE="12" \
  -e HUGGING_FACE_HUB_TOKEN=<hf_token> \
  <image_name>


# To see file structure in docker volume
sudo docker run --rm -it -v <volume_name>:/data alpine sh


# To delete docker volume
sudo docker volume rm <volume_name>


# To check logs in -d container
sudo docker logs <container_name_or_id>