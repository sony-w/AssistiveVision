# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         apt-utils \
         build-essential \
         ca-certificates \
         cmake \
         curl \
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
         wget

# upgrade pip so we can get the binary versions of pyarrow
RUN wget -q -O get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

# this installs the right pip and dependencies for the fresh python
RUN pip3 install wheel 
RUN pip3 install cython ipython

# python dependencies
RUN pip3 install \
        ninja \
        yacs \
        matplotlib \
        tqdm \
        opencv-python \
        shapely \
        scipy \
        tensorboardX \
        pyclipper \
        Polygon3 \
        editdistance 

# install PyTorch
RUN pip3 install \
        torch==1.7.1+cu101 \
        torchvision==0.8.2+cu101 \
        -f https://download.pytorch.org/whl/torch_stable.html

RUN mkdir -p /opt/mts

# install pycocotools
RUN cd /opt/mts && \
    git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    python3 setup.py build_ext install

# install apex
RUN cd /opt/mts && \
    git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    python3 setup.py install --cuda_ext --cpp_ext

# clone repo
RUN cd /opt/mts && \
    git clone https://github.com/MhLiao/MaskTextSpotterV3.git && \
    cd MaskTextSpotterV3 && \
    sed -i "s/if torch.cuda.is_available() and CUDA_HOME is not None/if True/g" setup.py && \
    sed -i "s/AT_CHECK/TORCH_CHECK/g" maskrcnn_benchmark/csrc/cuda/deform_pool_cuda.cu && \
    sed -i "s/AT_CHECK/TORCH_CHECK/g" maskrcnn_benchmark/csrc/cuda/deform_conv_cuda.cu && \
    python3 setup.py build develop

RUN mkdir -p /opt/mts/MaskTextSpotterV3/output/mixtrain
COPY trained_model.pth /opt/mts/MaskTextSpotterV3/output/mixtrain
COPY gatsby.jpg /opt/mts/MaskTextSpotterV3

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_APP=/opt/program/predictor.py

# Set up the program in the image
COPY predictor /opt/program
WORKDIR /opt/program