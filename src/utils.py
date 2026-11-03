import os
import random
import torch
import subprocess
import numpy as np
import pandas as pd


def set_seed(seed: int):
    """
    Set the seed value for reproducibility
    """
    # Set the seed
    random.seed(seed)
    torch.manual_seed(seed)

def read_fasta_file(fasta_file_path):

    identifier = None

    with open(fasta_file_path) as f:

        for line in f:
            if line[0] == ">":

                # If it is the line containing the identifier name, then return the previous pair
                if identifier is not None:
                    yield identifier, sequence.upper()

                # Set the identifier and initialize the sequence
                identifier = line[1:].strip()
                sequence = ""

            else:
                sequence += line.strip()

        # If the identifier is set, then return the last pair of the file
        if identifier is not None:
            yield identifier, sequence.upper()

def compute_coverage(bam_file_path:str, edge_len:int = 75):
    '''
    Arguments:
        bam_file_path: Input bam file path
        edge_len: It is used to strip off the first and last 'edge_len' bases when calculating coverage
    '''

    try:
        bed_file_process = subprocess.Popen(
            ['bedtools', 'genomecov', '-bga', '-ibam', bam_file_path],
            universal_newlines=True,
            stdout=subprocess.PIPE
        )
        # Note: -bga (the "a" stands for all) forces the output to include regions with even zero coverage while
        # -bg would only report regions that have at least 1 read mapped to them.

    except subprocess.CalledProcessError as e:
        raise Exception(f"Error while running bedtools: {e.stderr}")

    # Read the file
    df = pd.read_csv(
        bed_file_process.stdout,
        sep='\t',
        header=None,
        names=['contig_name', 'start', 'end', 'coverage'],
        dtype={'contig_name': str, 'start': int, 'end': int, 'coverage': int}
    )

    # A dictionary to store the resulting depth arrays for each contig
    contig_names, contig_depth_means, contig_depth_vars = [], [], []

    # Group the lines by the contig name
    for contig_name, group in df.groupby('contig_name'):

        # Calculate lengths of the regions. The sum of the values in each row of 'lengths' will be the contig length.
        lengths = group['end'] - group['start']
        # Get the coverage values.
        coverage_values = group['coverage']
        # Expand the values so the length of depth_value will be equal to contig length
        depth_value = np.repeat(coverage_values.values, lengths.values)
        # The ends of a contig usually have artificially low or highly skewed coverage, known as the "edge effect."
        # By stripping off the first and last 'edge_len' bases, we can discard this noisy, unreliable terminal data, so
        # it doesn't skew the mean and variance calculations.
        depth_value = depth_value[edge_len:-edge_len]
        # Add the contig name, depth mean and variance
        contig_names.append(contig_name)
        contig_depth_means.append(depth_value.mean())
        contig_depth_vars.append(depth_value.var())

    return contig_names, contig_depth_means, contig_depth_vars

def semibin2_calculate_coverage(depth_stream, bam_file, must_link_threshold, edge=75, is_combined=False, contig_threshold=1000, sep=None, sample_contig_threshold=None):
    """
    depth_stream : an iterable like the output of bedtools genomecov
    bam_file : str filename
    must_link_threshold : int
    edge : int
    is_combined : bool

    """
    import pandas as pd
    import numpy as np
    from itertools import groupby

    contigs = []
    mean_coverage = []
    var = []

    split_contigs = []
    split_coverage = []

    for contig_name, lines in groupby(depth_stream, lambda ell: ell.split('\t', 1)[0]):
        lengths = []
        values = []
        for line in lines:
            line_split = line.strip().split('\t')
            length = int(line_split[2]) - int(line_split[1])
            value = int(float(line_split[3]))
            lengths.append(length)
            values.append(value)
        depth_value = np.zeros(sum(lengths), dtype=int)
        s = 0
        for ell,v in zip(lengths, values):
            depth_value[s:s+ell] = v
            s += ell

        if sep is None:
            cov_threshold = contig_threshold
        else:
            try:
                sample, contig = contig_name.split(sep)
                cov_threshold = sample_contig_threshold[sample]
            except ValueError:
                raise ValueError(f"Error parsing contig name '{contig_name}' in file {bam_file} (trying to split by {sep} separator)")
            except KeyError:
                raise KeyError(f"Error: {sample} not found (parsing {bam_file}, the original contig name was {contig_name} and using {sep} as separator)")
        if len(depth_value) < cov_threshold:
            continue
        depth_value_ = depth_value[edge:-edge]
        mean_coverage.append(depth_value_.mean())
        var.append(depth_value_.var())
        contigs.append(contig_name)

        if is_combined:
            if len(depth_value) >= must_link_threshold:
                middle = len(depth_value_) // 2
                split_contigs.append(contig_name + '_1')
                split_contigs.append(contig_name + '_2')

                split_coverage.append(depth_value_[:middle].mean())
                split_coverage.append(depth_value_[middle:].mean())

    if is_combined:
        contig_cov = pd.DataFrame(
                { '{0}_cov'.format(bam_file): mean_coverage,
                }, index=contigs)
        split_contig_cov = pd.DataFrame(
                { '{0}_cov'.format(bam_file): split_coverage,
                }, index=split_contigs)
        return contig_cov, split_contig_cov
    else:
        return pd.DataFrame({
            '{0}_mean'.format(bam_file): mean_coverage,
            '{0}_var'.format(bam_file): var,
            }, index=contigs), None

def semibin2_generate_cov(bam_file: str, bam_index, out, threshold, is_combined, contig_threshold, logger, sep = None):
    """
    Call bedtools and generate coverage file

    bam_file: bam file used
    out: output
    threshold: threshold of contigs that will be binned
    is_combined: if using abundance feature in deep learning. True: use
    contig_threshold: threshold of contigs for must-link constraints
    sep: separator for multi-sample binning
    """
    import numpy as np
    logger.debug(f'Processing `{bam_file}`')
    bam_name = os.path.split(bam_file)[-1] + f'_{bam_index}'

    bed_p = subprocess.Popen(
        ['bedtools', 'genomecov',
         '-bga',
         '-ibam', bam_file],
        universal_newlines=True,
        stdout=subprocess.PIPE)

    contig_cov, must_link_contig_cov = semibin2_calculate_coverage(
                                    bed_p.stdout,
                                    bam_file,
                                    threshold,
                                    is_combined=is_combined,
                                    sep=sep,
                                    contig_threshold=(contig_threshold if sep is None else 1000),
                                    sample_contig_threshold=(contig_threshold if sep is not None else None))

    if bed_p.wait() != 0:
        raise OSError(f"Failure in running bedtools ({bam_file})")
    elif len(contig_cov) == 0:
        logger.critical(f"Running `bedtools genomecov` did not return an error, but the result is an empty file (processing {bam_file}). "
                        "Please check your input files: SemiBin expects that they are sorted BAM files.")
        raise OSError(f"Running bedtools returned an empty file ({bam_file})")

    contig_cov += 1e-5
    if not is_combined:
        with open(os.path.join(out, '{}_data_cov.csv'.format(bam_name)), 'w') as ofile:
            contig_cov.to_csv(ofile)
    else:
        if sep is None:
            abun_scale = (contig_cov.mean() / 100).apply(np.ceil) * 100
            contig_cov = contig_cov.div(abun_scale)
        with open(os.path.join(out, f'{bam_name}_data_cov.csv'), 'w') as ofile:
            contig_cov.to_csv(ofile)

        must_link_contig_cov = must_link_contig_cov.apply(lambda x: x + 1e-5)
        if sep is None:
            abun_split_scale = (must_link_contig_cov.mean() / 100).apply(np.ceil) * 100
            must_link_contig_cov = must_link_contig_cov.div(abun_split_scale)

        with open(os.path.join(out, f'{bam_name}_data_split_cov.csv'), 'w') as ofile:
            must_link_contig_cov.to_csv(ofile)

    return bam_file