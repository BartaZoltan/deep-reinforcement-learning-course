FROM python:3.11-slim

# Keep Python output unbuffered and avoid writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
		PYTHONUNBUFFERED=1 \
		PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# System dependencies:
# - build-essential: for compiling some Python wheels when needed
# - git: convenient for pulling/installing extras
# - libgl1, libglib2.0-0: common runtime deps for some gymnasium render backends
RUN apt-get update && apt-get install -y --no-install-recommends \
			build-essential \
			git \
			libgl1 \
			libglib2.0-0 \
		&& rm -rf /var/lib/apt/lists/*

# Install Python dependencies (kept in requirements.txt for students/Colab too)
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip \
	&& python -m pip install -r /workspace/requirements.txt

# Copy the course materials into the image (after deps for better layer caching)
COPY . /workspace

EXPOSE 8888

# Start JupyterLab with no token/password (intended for local use).
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.root_dir=/workspace"]
