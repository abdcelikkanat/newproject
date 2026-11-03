import numpy as np
from evaluation.markers import NUM_SINGLE_COPY_MARKERS


def score_bin(contig_markers: list):
    """
    Estimate the quality of a bin from the single-copy marker genes of its contigs
    Arguments:
        contig_markers: The sets of marker genes found on each contig of the bin
    Returns:
        The F1-like score, completeness and contamination of the bin
    """
    total = sum(len(markers) for markers in contig_markers)
    distinct = len(set().union(*contig_markers))
    if total == 0:
        return 0., 0., 0.

    completeness = distinct / NUM_SINGLE_COPY_MARKERS
    # Every extra copy of a single-copy marker gene is counted as contamination
    contamination = (total - distinct) / total
    score = 2 * completeness * (1 - contamination) / (completeness + 1 - contamination)

    return score, completeness, contamination

def get_best_bin(clusterings: dict, remaining: np.ndarray, contig_lens: np.ndarray, contig_markers: list,
                 bin_total_min_len: int, max_contaminations: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 1.)):
    """
    Find the best bin among the clusters of all clusterings, using only the remaining contigs
    Arguments:
        clusterings: A dictionary mapping each eps value to its cluster labels (one per contig), -1 means noise
        remaining: Boolean mask of the contigs that have not been assigned to a bin yet
        contig_lens: Contig lengths
        contig_markers: The sets of marker genes found on each contig
        bin_total_min_len: Minimum total length of a bin
        max_contaminations: Contamination limits, which are relaxed until a bin is found
    Returns:
        The contig indices of the best bin, or None if there is no bin satisfying the constraints
    """
    remaining_ids = np.flatnonzero(remaining)

    # Score all candidate bins once, then pick the best one under the loosest contamination limit needed
    candidates = []
    for labels in clusterings.values():
        labels = labels[remaining_ids]
        order = np.argsort(labels, kind="stable")
        cluster_labels, starts = np.unique(labels[order], return_index=True)
        for label, bin_ids in zip(cluster_labels, np.split(remaining_ids[order], starts[1:])):
            if label == -1:
                continue
            bin_len = contig_lens[bin_ids].sum()
            if bin_len < bin_total_min_len:
                continue
            score, _, contamination = score_bin([contig_markers[i] for i in bin_ids])
            if score > 0:
                candidates.append((score, contamination, bin_len, bin_ids))

    for max_contamination in max_contaminations:
        valid = [c for c in candidates if c[1] <= max_contamination]
        if valid:
            # Highest score first; for equal scores, the smaller bin
            return max(valid, key=lambda c: (c[0], -c[2]))[3]

    return None

def select_bins(clusterings: dict, contig_lens: np.ndarray, contig_markers: list, bin_total_min_len: int):
    """
    Greedily extract the best bins from the clusterings, removing the contigs of each selected bin from the rest
    Arguments:
        clusterings: A dictionary mapping each eps value to its cluster labels (one per contig), -1 means noise
        contig_lens: Contig lengths
        contig_markers: The sets of marker genes found on each contig
        bin_total_min_len: Minimum total length of a bin
    Returns:
        A list of the contig indices of each bin
    """
    remaining = np.ones(len(contig_lens), dtype=bool)
    bins = []
    while remaining.sum() > 1 and contig_lens[remaining].sum() >= bin_total_min_len:
        best_bin = get_best_bin(clusterings, remaining, contig_lens, contig_markers, bin_total_min_len)
        if best_bin is None:
            break
        bins.append(best_bin)
        remaining[best_bin] = False

    return bins
