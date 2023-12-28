FROM --platform=linux/amd64 nvcr.io/nvidia/cuda:12.3.1-runtime-ubuntu22.04

WORKDIR /clearquote

RUN apt update && apt install -y python3 python3-pip
RUN pip3 install --upgrade pip

COPY requirements/requirements-cpu.txt ./requirements-cpu.txt
RUN pip3 install -r ./requirements-cpu.txt

COPY test_predict.py ./test_predict.py

CMD ["bash"]
