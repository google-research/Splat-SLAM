FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
# 8.0+PTX means all architectures with compute capability >= 8.0 are supported.
# This includes all consumer GPUs from the Ampere (RTX 30xx) and Ada Lovelace (RTX 40xx) families.
# If you want to support other GPUs, consult https://developer.nvidia.com/cuda-gpus
ENV TORCH_CUDA_ARCH_LIST="8.0+PTX" 

RUN apt update && \
    apt install -y --no-install-recommends tzdata git unzip wget libglib2.0-0 libgl1-mesa-glx && \
    apt clean && rm -rf /var/lib/apt/lists/*
    

WORKDIR /workspace
RUN git clone https://github.com/google-research/Splat-SLAM.git --recursive

WORKDIR /workspace/Splat-SLAM

RUN sed -i 's/p_view\.z <= 0\.2f/p_view\.z <= 0\.001f/' /workspace/Splat-SLAM/thirdparty/diff-gaussian-rasterization-w-pose/cuda_rasterizer/auxiliary.h && \
    pip install --no-cache-dir -e thirdparty/lietorch/ \
                                   thirdparty/diff-gaussian-rasterization-w-pose/ \
                                   thirdparty/simple-knn/ \
                                   thirdparty/evaluate_3d_reconstruction_lib/ && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytorch-lightning==1.9 --no-deps && \
    pip install --no-cache-dir gdown

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y --no-install-recommends tzdata git unzip wget libglib2.0-0 libgl1-mesa-glx && \
    apt clean && rm -rf /var/lib/apt/lists/*
    
COPY --from=builder /workspace/Splat-SLAM /workspace/Splat-SLAM
COPY --from=builder /opt/conda /opt/conda
COPY scripts/download_pretrained_model.sh /usr/local/bin/download_pretrained_model.sh
RUN chmod +x /usr/local/bin/download_pretrained_model.sh

WORKDIR /workspace/Splat-SLAM

CMD ["tail", "-f", "/dev/null"]