# Dockerfile for DensePose Injury Detection Service
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Detectron2 and DensePose
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install opencv-python pillow numpy pyyaml requests fastapi uvicorn
RUN pip install langchain langchain-google-genai google-generativeai
RUN pip install git+https://github.com/facebookresearch/detectron2.git
RUN pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

# Copy app code
WORKDIR /app
COPY imageprocessing_denseposev3_langchain.py /app/

# Expose port
EXPOSE 8000

# Entrypoint
CMD ["uvicorn", "imageprocessing_denseposev3_langchain:app", "--host", "0.0.0.0", "--port", "8000"]
