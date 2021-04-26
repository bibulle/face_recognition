# This is a sample Dockerfile you can modify to deploy your own app based on face_recognition

FROM python:3-slim

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS


# The rest of this file just runs an example script.

# If you wanted to use this Dockerfile to run your own app instead, maybe you would do this:
# COPY . /root/your_app_or_whatever
# RUN cd /root/your_app_or_whatever && \
#     pip3 install -r requirements.txt
# RUN whatever_command_you_run_to_start_your_app

COPY requirements.txt /root/face_recognition/
RUN cd /root/face_recognition && \
    pip3 --default-timeout=1000 install -r requirements.txt

COPY . /root/face_recognition
RUN cd /root/face_recognition && \
    pip3 --default-timeout=1000 install -r requirements.txt && \
    python3 setup.py install

VOLUME ["/train_dir"]
VOLUME ["/test_dir"]

#RUN cd /root/face_recognition && \
    #mkdir -p /train_dir/obama && \
    #mkdir -p /train_dir/biden && \
    #cp examples/obama_small.jpg /train_dir/obama/ && \
    #cp examples/biden.jpg /train_dir/biden/ && \
#    cp examples/obama2.jpg test_image.jpg

CMD cd /root/face_recognition && \
    python3 face_recognition_knn.py
