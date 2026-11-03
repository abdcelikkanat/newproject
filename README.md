## UncertainGen Adaptation for SemiBin2 Implementation

#### TODO List
- Check SummaryWriter can be also used for saving the output logs

### Setting up the environment
The evaluation needs three command-line tools besides the Python packages: `bedtools` (coverage), `prodigal` (gene
prediction) and `hmmer` (marker gene search). They are not available on pip, so we use either conda or Docker.

#### Conda
The `environment.yml` file installs Python, the command-line tools (from bioconda) and the packages in `requirements.txt`:
```
conda env create -f environment.yml
conda activate mysemibin
```
PyTorch from the default pip index ships with GPU support on Linux (CUDA) and macOS (MPS). On a Linux machine
without a GPU, you can avoid the large CUDA download by adding the CPU-only wheel index of PyTorch while creating the
environment, so that pip picks the CPU build (`torch==2.14.1+cpu`):
```
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu conda env create -f environment.yml
```

#### Docker
The image is CPU-only by default. For NVIDIA GPUs, pass the CUDA wheel index of PyTorch while building
(see https://pytorch.org/get-started/locally for the available CUDA versions):
```
docker build -t mysemibin .                                                                   # CPU
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 -t mysemibin . # GPU
```
The image contains the code and the tools, but not the data: `dataset`, `models` and `output` are left out by
`.dockerignore`. Mounting the root folder makes them available, keeps the results on the host and lets you edit the
code without rebuilding the image (rebuild only after changing `requirements.txt` or the `Dockerfile`):
```
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)":/workdir -it mysemibin bash            # Open a shell
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)":/workdir --gpus all -it mysemibin bash # Open a shell, GPU build
```
Running the image without a command prints the arguments of the evaluation script. See [Evaluation](#evaluation) for
a full command.

On Docker Desktop, give the virtual machine enough memory (Settings > Resources), since the evaluation loads the whole
assembly and the coverage of the BAM file.

Without Docker Desktop, [Docker Sandboxes](https://docs.docker.com/ai/sandboxes/) (`sbx`) provides a Docker daemon in a
microVM. Its network policy has to allow the PyTorch hosts for the build:
```
sbx create --name mysemibin --memory 16g --cpus 6 shell .
sbx policy allow network download.pytorch.org:443,download-r2.pytorch.org:443 --sandbox mysemibin
sbx exec -it mysemibin bash
```
The last command opens a shell inside the sandbox. There, go to the root folder (it has the same absolute path as on
the host) and use the `docker` commands in this file as they are.

### Datasets
We first need the ```gdown``` package to get the data. 
```
pip install gdown
```

Then, please run the following commands to download the training dataset:
```
gdown 1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp
unzip dnabert-s_train.zip
```
and the evaluation datasets can be retrieved similarly.
```
gdown 1I44T2alXrtXPZrhkuca6QP3tFHxDW98c
unzip dnabert-s_eval.zip
```

### Training
You can train the model by setting the required parameters. An example is given below:
```
python src/vib.py --input DATA_FOLDER_PATH/train_2m.csv --output EMB_FILE_PATH --max_read_num 100 --device cpu
```
Note that the training set contains two million sequence pairs, but you can randomly sample desired amount of sequences
by setting the ```max_read_num``` parameter. 

For more details, you can run the following command:
```
python src/vib.py --help
```

### Evaluation
The evaluation script follows the long-read binning of SemiBin2: it combines the contig embeddings with the log mean
coverage, runs DBSCAN for several `eps` values, and greedily selects the bins using the single-copy marker genes.
Run it from the root folder, passing the model, the contigs, the BAM file and the output folder:
```
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)":/workdir mysemibin \
    python -m evaluation.evaluation \
    --model_path models/test.model \
    --contig_file_path dataset/PaPr00000116MP_contigs.fasta \
    --bam_file_path dataset/PaPr00000116MP_1-NP.bam \
    --output_dir output
```
Inside the conda environment, run the same `python -m evaluation.evaluation` command directly. The other arguments
have default values; add `--help` to list them.

The output folder contains the selected bins (`bins/*.fa`), the bin of each contig (`contig_bins.tsv`) and the
intermediate marker gene files.

The marker gene profiles in `evaluation/data/marker.hmm` (107 bacterial single-copy genes from TIGRFAM) are taken from
[SemiBin2](https://github.com/BigDataBiology/SemiBin) (MIT license).
