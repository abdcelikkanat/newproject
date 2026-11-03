import os
import subprocess
import tempfile


# TIGRFAM profiles of 107 bacterial single-copy marker genes, taken from SemiBin2 (MIT license)
MARKER_HMM_PATH = os.path.join(os.path.dirname(__file__), "data", "marker.hmm")
# Some profiles in marker.hmm describe the same gene, so they are merged (111 profiles -> 107 marker genes)
_MERGED_MARKERS = {"TIGR00388": "TIGR00389", "TIGR00471": "TIGR00472", "TIGR00408": "TIGR00409", "TIGR02386": "TIGR02387"}
NUM_SINGLE_COPY_MARKERS = 107


def run_prodigal(contigs: dict, output_faa_path: str, num_process: int = 1):
    """
    Predict the genes of the contigs with prodigal, running one process per chunk of the contigs
    Arguments:
        contigs: A dictionary mapping the contig names to their sequences
        output_faa_path: Output file path for the predicted protein sequences
        num_process: Number of prodigal processes to run in parallel
    """
    # Split the contigs into chunks of similar total length, assigning the longest contigs first to the lightest chunk
    chunks, chunk_lens = [[] for _ in range(num_process)], [0] * num_process
    for contig_name in sorted(contigs, key=lambda name: len(contigs[name]), reverse=True):
        lightest = chunk_lens.index(min(chunk_lens))
        chunks[lightest].append(contig_name)
        chunk_lens[lightest] += len(contigs[contig_name])

    with tempfile.TemporaryDirectory() as temp_dir:

        # Write each chunk into a separate fasta file and start a prodigal process for it
        processes, faa_paths = [], []
        for i, chunk in enumerate(chunk for chunk in chunks if chunk):
            fasta_path, faa_path = os.path.join(temp_dir, f"chunk_{i}.fna"), os.path.join(temp_dir, f"chunk_{i}.faa")
            with open(fasta_path, "w") as f:
                for contig_name in chunk:
                    f.write(f">{contig_name}\n{contigs[contig_name]}\n")
            # -p meta: gene prediction for the metagenomic sequences, -q: quiet mode, -o: the gene coordinates (unused)
            processes.append(subprocess.Popen(
                ["prodigal", "-i", fasta_path, "-a", faa_path, "-p", "meta", "-q", "-o", os.devnull], 
                stderr=subprocess.PIPE, universal_newlines=True
            ))
            faa_paths.append(faa_path)

        # Wait for all processes to finish
        for process in processes:
            _, stderr = process.communicate()
            if process.returncode != 0:
                raise Exception(f"Error while running prodigal: {stderr}")

        # Merge the predicted protein sequences of the chunks
        with open(output_faa_path, "w") as output_file:
            for faa_path in faa_paths:
                with open(faa_path) as f:
                    output_file.write(f.read())

def run_hmmsearch(faa_path: str, output_domtblout_path: str, num_process: int = 1, hmm_path: str = MARKER_HMM_PATH):
    """
    Search the marker gene profiles in the predicted protein sequences
    Arguments:
        faa_path: Input file path of the predicted protein sequences
        output_domtblout_path: Output file path of the per-domain hits table
        num_process: Number of CPUs used by hmmsearch
        hmm_path: Input file path of the marker gene profiles
    """
    # --cut_tc: use the trusted cutoffs of the profiles, --noali/-o: skip the human-readable output
    process = subprocess.run(
        ["hmmsearch", "--domtblout", output_domtblout_path, "--cut_tc", "--cpu", str(num_process), "--noali",
         "-o", os.devnull, hmm_path, faa_path],
        stderr=subprocess.PIPE, universal_newlines=True
    )
    if process.returncode != 0:
        raise Exception(f"Error while running hmmsearch: {process.stderr}")

def parse_hmmsearch(domtblout_path: str, min_coverage: float = 0.4):
    """
    Get the marker genes found on each contig from the hmmsearch output
    Arguments:
        domtblout_path: Input file path of the per-domain hits table
        min_coverage: Minimum fraction of the profile that a hit must cover
    Returns:
        A dictionary mapping the contig names to the set of marker genes found on them
    """
    contig2markers = dict()
    with open(domtblout_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            # The columns: 0 the ORF name, 3 the profile name, 5 the profile length, 15-16 the profile alignment span
            fields = line.split()
            orf_name, marker, marker_len = fields[0], fields[3], int(fields[5])
            marker_start, marker_end = int(fields[15]), int(fields[16])
            # Skip the hits covering only a small part of the profile
            if (marker_end - marker_start) / marker_len <= min_coverage:
                continue
            # Prodigal names the ORFs as "<contig name>_<ORF index>"
            contig_name = orf_name.rsplit("_", 1)[0]
            contig2markers.setdefault(contig_name, set()).add(_MERGED_MARKERS.get(marker, marker))

    return contig2markers

def get_contig_markers(contigs: dict, output_dir: str, num_process: int = -1):
    """
    Find the single-copy marker genes on each contig
    Arguments:
        contigs: A dictionary mapping the contig names to their sequences
        output_dir: Output directory for the predicted protein sequences and the hmmsearch hits
        num_process: Number of processes used by prodigal and hmmsearch
    Returns:
        A dictionary mapping the contig names to the set of marker genes found on them
    """
    if num_process == -1:
        num_process = os.cpu_count()

    faa_path, domtblout_path = os.path.join(output_dir, "genes.faa"), os.path.join(output_dir, "markers.domtblout")
    run_prodigal(contigs=contigs, output_faa_path=faa_path, num_process=num_process)
    run_hmmsearch(faa_path=faa_path, output_domtblout_path=domtblout_path, num_process=num_process)

    return parse_hmmsearch(domtblout_path=domtblout_path)
