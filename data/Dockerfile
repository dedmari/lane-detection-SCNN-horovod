FROM ubuntu:18.04

ARG python=3.7
ENV PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-7 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        libgl1-mesa-glx \
        libglib2.0-0 \
        unzip

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py


# Install other packages
RUN pip install numpy
RUN pip install opencv-python
RUN pip install ujson

# Copy code to docker image
RUN mkdir /workdir
COPY ./gen_seg_label.py /workdir/
COPY ./entrypoint.sh /workdir/

# Change working directory
WORKDIR /workdir

# Making entry file executable
RUN chmod +x ./entrypoint.sh

# Running the script
ENTRYPOINT ["./entrypoint.sh"]