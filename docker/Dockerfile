FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /usr/src/app

COPY . .

RUN apt-get clean

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir \
        scikit-image \
        scikit-learn \
        scipy \
        ipython \
        pandas \
        packaging \
        imageio \
        pyyaml \
        seaborn \
        opencv-python \
        pillow

RUN apt update && apt install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

LABEL maintainer="Ian Mason <imason@mit.edu>"