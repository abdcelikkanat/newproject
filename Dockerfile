FROM python:3.12-slim

# The PyTorch wheel index: CPU-only by default. For NVIDIA GPUs, build with e.g.
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 -t newproject .
# and run the container with --gpus all
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Command-line tools used by the evaluation (over HTTPS, since Docker Sandboxes allows only port 443 for the mirror)
RUN sed -i 's#http://deb.debian.org#https://deb.debian.org#' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends bedtools prodigal hmmer \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workdir

COPY requirements.txt .

# Install the pinned torch version from the selected index first, then the remaining packages
RUN pip install --no-cache-dir "$(grep '^torch==' requirements.txt)" --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Show the progress messages immediately in the container logs
ENV PYTHONUNBUFFERED=1

# The evaluation needs its input paths, so show the available arguments by default (see README.md for a full command)
CMD ["python", "-m", "evaluation.evaluation", "--help"]
