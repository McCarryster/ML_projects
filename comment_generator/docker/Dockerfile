# Use an official PyTorch image with CUDA 11.8 support
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime  

# Set working directory
WORKDIR /app

# Copy requirements file
COPY docker/requirements.txt .

# Install dependencies
RUN pip install --verbose -r requirements.txt

# Ensure torchvision is installed with CUDA support
RUN pip install torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Install transformers library
RUN pip install transformers --upgrade

# Copy the model
COPY models/llama3.1/vikhr_8B_instruct/ ./model/

# Copy the data
COPY data/data_types/processed_data/filtered/comment_only_train_set.json ./data/
COPY data/data_types/processed_data/filtered/comment_only_val_set.json ./data/

# Copy the scripts
COPY scripts/fine_tune_scripts/ fine_tune_scripts/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the training script
CMD ["python", "./fine_tune_scripts/training_loop.py"]