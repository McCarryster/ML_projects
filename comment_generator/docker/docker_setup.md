1. To connect to the server:
    - ssh -i /path/to/key/key-name.pem ubuntu@00.00.000.0


2. Docker and cuda setup:
    - sudo apt update && sudo apt install ubuntu-drivers-common && ubuntu-drivers devices && sudo ubuntu-drivers autoinstall && sudo reboot
    - Run those commands https://docs.docker.com/engine/install/ubuntu/
    - Now run those commands https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html


3. Login to docker on a server
    - sudo docker login


4. Pull image
    - sudo docker pull <image_name>