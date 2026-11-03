#!/usr/bin/env bash
# Build the Apptainer/Singularity image from the definition file. Run it from the repo root.
#
# The #SBATCH lines hold only options that work on any Slurm cluster. Pass the cluster-specific ones
# (partition, account, time limit) to sbatch, since command-line options override these lines:
#     sbatch --partition=cpu sh/build_container.sh      # AI-Cloud
#     sbatch --partition=<name> sh/build_container.sh   # another Slurm cluster
#     bash sh/build_container.sh                        # without Slurm, e.g. a workstation
#
# Arguments (optional): the definition file and the output image
#     sbatch --partition=cpu sh/build_container.sh newproject.def newproject.sif
#
#SBATCH --job-name=build_newproject
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G

set -eu

input_def=${1:-newproject.def}
output_sif=${2:-${input_def%.def}.sif}

# Slurm runs a copy of this script, so resolve relative paths from the submit folder
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ ! -f "$input_def" ]; then
    echo "Definition file '$input_def' not found in $PWD. Submit from the repo root, e.g." >&2
    echo "    cd <repo root> && sbatch --partition=cpu sh/build_container.sh" >&2
    exit 1
fi
if [ -d "$output_sif" ]; then
    echo "The output '$output_sif' is a folder; give an image file name such as newproject.sif." >&2
    exit 1
fi

# Newer clusters provide apptainer, older ones singularity (both read the same definition file)
if command -v apptainer > /dev/null; then
    runtime=apptainer
elif command -v singularity > /dev/null; then
    runtime=singularity
else
    echo "Neither apptainer nor singularity was found (on some clusters, try 'module load apptainer')." >&2
    exit 1
fi

# Keep the large temporary and cache files out of /tmp. Set CONTAINER_BUILD_DIR to use another folder,
# e.g. a scratch disk when the home folder has a small quota.
build_dir=${CONTAINER_BUILD_DIR:-$HOME/.singularity}
export APPTAINER_TMPDIR=$build_dir/tmp SINGULARITY_TMPDIR=$build_dir/tmp
export APPTAINER_CACHEDIR=$build_dir/cache SINGULARITY_CACHEDIR=$build_dir/cache
mkdir -p "$build_dir/tmp" "$build_dir/cache"

# Root does not need --fakeroot. Set BUILD_FLAGS to override, e.g. BUILD_FLAGS=--remote.
if [ "$(id -u)" -eq 0 ]; then
    default_flags=""
else
    default_flags="--fakeroot"
fi

echo "Building $output_sif from $input_def with $runtime"
# shellcheck disable=SC2086  # BUILD_FLAGS may hold several flags
$runtime build ${BUILD_FLAGS-$default_flags} "$output_sif" "$input_def"
