# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.
ARG CUDA="10.1"
ARG CUDNN="7"

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         apt-utils \
         build-essential \
         ca-certificates \
         cmake \
         ffmpeg \
         git \
         gnupg \
         libre2-dev \
         libsm6 \
         libxext6 \
         lsb-release \
         nginx \
         pkg-config \
         python3-pip \
         python3-setuptools \
         python3-dev \
         wget \
    && rm -rf /var/lib/apt/lists/*

# upgrade pip so we can get the binary versions of pyarrow
RUN wget -q -O get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

# Download models and data
RUN mkdir -p /content/model_data
RUN wget -q -O  /content/model_data/answers_lorra.txt https://dl.fbaipublicfiles.com/pythia/data/answers_textvqa_more_than_1.txt
RUN wget -q -O  /content/model_data/vocabulary_100k.txt https://dl.fbaipublicfiles.com/pythia/data/vocabulary_100k.txt
RUN wget -q -O  /content/model_data/detectron_model.pth  https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth 
RUN wget -q -O  /content/model_data/lorra.pth https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/lorra_best.pth
RUN wget -q -O  /content/model_data/lorra.yaml https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/lorra.yml
RUN wget -q -O  /content/model_data/detectron_model.yaml https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml
RUN wget -q -O  /content/model_data/detectron_weights.tar.gz https://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
RUN wget -q -O  /content/model_data/wiki.en.bin https://dl.fbaipublicfiles.com/pythia/pretrained_models/fasttext/wiki.en.bin
RUN tar xf /content/model_data/detectron_weights.tar.gz

# Install Apache Arrow
# RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && \
#     apt-get install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && \
#     apt-get update && \
#     apt-get install -y -V libarrow-dev libarrow-python-dev

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN pip3 --no-cache-dir install \
        wheel \
        cython

RUN pip3 --no-cache-dir install \
        numpy \
        opencv-python \
        pandas \
        pyarrow \
        scipy \
        scikit-learn \
        flask \
        gunicorn \
        ipython \
        ninja \
        yacs \
        matplotlib \
        demjson

RUN pip3 install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI

RUN cd /content/ && \
    rm -rf fastText && \
    git clone https://github.com/facebookresearch/fastText.git fastText && \
    cd /content/fastText && \
    pip3 install . && \
    make

RUN cd /content/ && \
    rm -rf pythia && \
    git clone --branch v0.3.1 https://github.com/facebookresearch/pythia.git pythia && \
    cd /content/pythia && \
    sed -i 's/==/>=/g' requirements.txt && \
    pip3 install -e .

RUN mkdir -p /content/pythia/pythia/.vector_cache
RUN mv /content/model_data/wiki.en.bin /content/pythia/pythia/.vector_cache/wiki.en.bin

RUN apt-get update && \
    apt-get -y install curl

# Install maskrcnn-benchmark to extract detectron features
# Compile custom layers and build mask-rcnn backbone
RUN cd /content && \
    git clone https://gitlab.com/meetshah1995/vqa-maskrcnn-benchmark.git && \
    cd /content/vqa-maskrcnn-benchmark && \
    python3 setup.py build && \
    python3 setup.py develop

RUN pip3 install google-api-python-client boto3

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:/content/fastText/python:/content/pythia:/content/vqa-maskrcnn-benchmark:${PATH}"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_APP=/opt/program/predictor.py

# extract detectron weights
RUN cd /content && \
    tar xvf /content/model_data/detectron_weights.tar.gz

# hack needed to update maskrcnn_benchmark to work against newer version of pytorch
RUN sed -i 's/PY3/PY37/g' /content/vqa-maskrcnn-benchmark/maskrcnn_benchmark/utils/imports.py

# Set up the program in the image
COPY lorra /opt/program
WORKDIR /opt/program