FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/miniforge3/bin:$PATH" \
    TORCH_CUDA_ARCH_LIST="8.0+PTX"

RUN apt update && \
    apt install -y --no-install-recommends tzdata git unzip wget libglib2.0-0 libgl1-mesa-glx && \
    wget -O Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3.sh -b -p /opt/miniforge3 && \
    rm Miniforge3.sh && \
    apt clean && rm -rf /var/lib/apt/lists/*
    

WORKDIR /workspace
RUN git clone https://github.com/google-research/Splat-SLAM.git --recursive

WORKDIR /workspace/Splat-SLAM
RUN conda create -n splat-slam python=3.10 && \
    conda clean -a -y

SHELL ["conda", "run", "-n", "splat-slam", "/bin/bash", "-c"]

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    sed -i 's/p_view\.z <= 0\.2f/p_view\.z <= 0\.001f/' /workspace/Splat-SLAM/thirdparty/diff-gaussian-rasterization-w-pose/cuda_rasterizer/auxiliary.h && \
    pip install --no-cache-dir -e thirdparty/lietorch/ \
                                   thirdparty/diff-gaussian-rasterization-w-pose/ \
                                   thirdparty/simple-knn/ \
                                   thirdparty/evaluate_3d_reconstruction_lib/ && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytorch-lightning==1.9 --no-deps && \
    pip install --no-cache-dir gdown

RUN conda init bash && \
    echo "conda activate splat-slam" >> ~/.bashrc

COPY download_pretrained_model.sh /usr/local/bin/download_pretrained_model.sh
RUN chmod +x /usr/local/bin/download_pretrained_model.sh

CMD ["tail", "-f", "/dev/null"]