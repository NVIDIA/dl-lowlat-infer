# syntax = docker/dockerfile:1.5
ARG NGC_IMAGE=nvcr.io/nvidia/cuda
ARG NGC_TAG=13.0.2-cudnn-devel-ubuntu24.04
ARG APP_DIR=/app/dl-lowlat-infer

FROM ${NGC_IMAGE}:${NGC_TAG} as develop

RUN apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --yes --quiet \
        curl \
        git \
        git-lfs \
        gdbserver \
        htop \
        numactl \
        openssh-client \
        python3-dev \
        python3-venv \
        python3-pip \
        rsync \
        sudo \
        cmake \
        gdb \
        libnuma-dev \
        vim \
        software-properties-common \
        kmod \
 && rm -rf /var/lib/apt/lists/*

ARG APP_DIR
WORKDIR ${APP_DIR}

ENV PATH="/app/python-venv/bin:${PATH}"

RUN --mount=type=cache,target=/root/.cache/pip \ 
    python3 -m venv /app/python-venv \
 && pip3 install --no-cache-dir --upgrade pip setuptools \
 && pip3 install --no-cache-dir \
    numpy \
    onnx \
    nvidia-cuda-runtime \
 && if [ "$(uname -m)" != "aarch64" ] && [ "$(uname -m)" != "arm64" ]; then \
        pip3 install --no-cache-dir onnxruntime-gpu; \
    fi

# Setup sudo accounts
COPY docker/files/sudo_nvidia /etc/sudoers.d/
COPY docker/files/90-profile.sh /opt/nvidia/entrypoint.d/

FROM develop as cmake

ARG SRC_DIR=/src/dl-lowlat-infer
WORKDIR ${SRC_DIR}

COPY cmake cmake
COPY CMakeLists.txt ./
COPY thirdparty thirdparty
COPY src src
COPY test test

ARG APP_DIR
ARG CMAKE_BUILD_TYPE=RELEASE
ARG CPU_ARCH=native
ARG CUDA_ARCHS="90a-real;100-real"
ARG BUILD_TESTS=OFF
RUN mkdir build \
 && cd build \
 && cmake \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DCPU_ARCH=${CPU_ARCH} \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
        -DCMAKE_INSTALL_PREFIX=${APP_DIR} \
        -DBUILD_TESTS=${BUILD_TESTS} \
        .. \
 && make -j$(nproc) install \
 && chmod -R a+w .

FROM develop as models

ARG APP_DIR
WORKDIR ${APP_DIR}

COPY src/python/generate_model.py \
     src/python/generate_data.py \
     src/python/requirements.txt \
     ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

WORKDIR ${APP_DIR}/data

# Generate weights for the models
RUN python3 ../generate_model.py --num-steps=64  --num-units=96  --num-inputs=128  --num-layers=2 --model=lstm_s --seed=42 --output-dir=. \
 && python3 ../generate_model.py --num-steps=96  --num-units=160 --num-inputs=192  --num-layers=3 --model=lstm_m --seed=42 --output-dir=. \
 && python3 ../generate_model.py --num-steps=128 --num-units=736 --num-inputs=512  --num-layers=4 --model=lstm_l --seed=42 --output-dir=.

# Note: Ping pong model weights are generated internally when using --ping-pong flag (no pre-generation needed)

# Generate data
RUN python3 ../generate_data.py --num-inputs=128 --seed=42 -o ./lstm_s.npy \
 && python3 ../generate_data.py --num-inputs=192 --seed=42 -o ./lstm_m.npy \
 && python3 ../generate_data.py --num-inputs=512 --seed=42 -o ./lstm_l.npy


FROM models as release

ARG GDRCOPY_DIR=/app/gdrcopy
COPY --from=cmake ${APP_DIR}/../gdrcopy ${GDRCOPY_DIR}/
# Make GDRCOPY_DIR available at runtime (inherited by models and release stages)
ENV PATH="${GDRCOPY_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${GDRCOPY_DIR}/lib:${LD_LIBRARY_PATH}"

ARG APP_DIR
WORKDIR ${APP_DIR}
COPY --from=cmake ${APP_DIR}/ ./
