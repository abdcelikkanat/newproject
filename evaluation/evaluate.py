import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from src.model import Model
from src.tokenizer import KmerTokenizer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph, sort_graph_by_row_values
from src.utils import read_fasta_file, compute_coverage
from evaluation import markers
from evaluation import bin_selection





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument(
        '--model_path', type=str, help='Model file path', required=True
    )
    parser.add_argument(
        '--contig_file_path', type=str, help='Contig file path', required=True,
    )
    parser.add_argument(
        '--bam_file_path', type=str, help="Bam file path", required=True,
    )
    parser.add_argument(
        '--output_dir', type=str, help='Output directory', required=True
    )
    parser.add_argument(
        '--k', type=int, default=4, help='k value'
    )
    parser.add_argument(
        '--kmer_counting_method', type=str, default="canonical", help='The approach to design k-mer features'
    )
    parser.add_argument(
        '--min_contig_len', type=int, default=1000, help='Length of minimum contig length'
    )
    parser.add_argument(
        '--edge_len', type=int, default=75, help='Edge length to trim contigs while computing coverage'
    )
    parser.add_argument(
        '--dbscan_max_neighbors', type=int, default=200, help='Maximum number of neighbors to use for DBSCAN'
    )
    parser.add_argument(
        '--dbscan_min_samples', type=int, default=5, help='Minimum number of samples to use for DBSCAN'
    )
    parser.add_argument(
        '--bin_total_min_len', type=int, default=200_000, help='Total minimum length of bin'
    )

    args = parser.parse_args()

    model_path = args.model_path
    contig_file_path = args.contig_file_path
    bam_file_path = args.bam_file_path
    output_dir = args.output_dir
    k = args.k
    kmer_counting_method = args.kmer_counting_method
    # Constraints
    min_contig_len = args.min_contig_len  # SemiBin2 default; contigs shorter than this are not binned
    edge_len = args.edge_len
    # DBSCAN parameters
    dbscan_max_neighbors = args.dbscan_max_neighbors
    dbscan_min_samples = args.dbscan_min_samples
    dbscan_eps_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    # Bin selection parameters
    bin_total_min_len = args.bin_total_min_len

    # If the output directory doesn't exist, create it
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    print(f"+ Loading model from {model_path}")
    kwargs, model_state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model = Model(**kwargs)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Define the tokenizer
    print(f"+ Defining tokenizer (counting method={kmer_counting_method})")
    tokenizer = KmerTokenizer(k=k, counting_method=kmer_counting_method)

    # Get the contig names and embeddings
    print(f"+ Reading the input fasta file and getting the embeddings of contigs.")
    init_time = time.time()
    contig_names, contig_lens, contig_embs, contig_seqs = [], [], [], dict()
    with torch.no_grad():
        for contig_name, seq in read_fasta_file(fasta_file_path=contig_file_path):

            # Skip the short contigs
            if len(seq) < min_contig_len:
                continue

            contig_names.append(contig_name)
            contig_lens.append(len(seq))
            contig_seqs[contig_name] = seq
            kmer_feature = torch.from_numpy(tokenizer.encode(seq)).to(torch.float).unsqueeze(0)
            contig_embs.append(model(kmer_feature))

    contig_embs = torch.cat(contig_embs, dim=0)
    contig_lens = np.asarray(contig_lens)
    # Construct a pandas data frame for the embeddings
    contig_embs_df = pd.DataFrame(data=contig_embs.numpy(), index=contig_names,
                                  columns=[f"Dim_{d}" for d in range(contig_embs.shape[1])])
    print(f"\t- Done in {time.time() - init_time} seconds.")

    # Get the coverage features, indexed by the contig name
    print(f"+ Reading the bam file to calculating coverage values.")
    init_time = time.time()
    bam_contig_names, contig_depth_means, contig_depth_vars = compute_coverage(bam_file_path=bam_file_path,
                                                                               edge_len=edge_len)
    depth_df = pd.DataFrame(data={"mean": np.log(contig_depth_means), "var": np.log(contig_depth_vars)},
                            index=bam_contig_names)
    print(f"\t- Done in {time.time() - init_time} seconds.")

    # Match the features by the contig name: keep only contigs present in all inputs, in the FASTA order
    print(f"+ Contigs in the fasta file and the bam file are being matching.")
    init_time = time.time()
    embs = contig_embs_df.join(depth_df, how="inner")

    # Check whether there is any contig in the given fasta but doesn't appear in the bam file
    missing = contig_embs_df.index.difference(embs.index)
    if len(missing):
        raise ValueError(f"{len(missing)} contigs have no coverage information!")

    # Filter out the contigs with zero coverage
    # (the mean coverage is already log-transformed, so zero coverage appears as -inf)
    zero_cov_mask = np.isneginf(embs["mean"].to_numpy())
    if zero_cov_mask.all():
        raise ValueError("All contigs have zero coverage!")
    if zero_cov_mask.any():
        print(f"\t- {zero_cov_mask.sum()} contigs with zero coverage are filtered out.")
        for contig_name in embs.index[zero_cov_mask]:
            del contig_seqs[contig_name]
        embs = embs[~zero_cov_mask]
        contig_embs_df = contig_embs_df[~zero_cov_mask]
        contig_lens = contig_lens[~zero_cov_mask]
        contig_names = embs.index.tolist()
    print(f"\t- Done in {time.time() - init_time} seconds.")

    # Build the clustering features
    print(f"+ Clustering started.")
    print(f"\t- Features for clustering is being constructed.")
    init_time = time.time()
    clustering_features = np.concatenate((contig_embs_df.to_numpy(), embs[["mean"]].to_numpy()), axis=1)

    # Build a sparse k-nearest neighbor distance graph so that DBSCAN does not need the full distance matrix
    print(f"\t- kNN distance graph is being constructed.")
    dist_matrix = kneighbors_graph(
        clustering_features, n_neighbors=min(dbscan_max_neighbors, clustering_features.shape[0] - 1),
        mode="distance", p=2,
        n_jobs=-1  # use all jobs
    )
    # Sort the neighbors of each row by distance once, instead of DBSCAN doing it for every eps value
    dist_matrix = sort_graph_by_row_values(dist_matrix, warn_when_not_sorted=False)

    # Run DBSCAN for each eps value, weighting each contig by its length
    print(f"\t- Running DBSCAN for different epsilon values.")
    clusterings = dict()
    for eps in dbscan_eps_values:
        print(f"\t\t- Current epsilon={eps}")
        labels = DBSCAN(eps=eps, min_samples=dbscan_min_samples, metric="precomputed", n_jobs=-1).fit(
            dist_matrix, sample_weight=contig_lens
        ).labels_
        clusterings[eps] = labels
        print(f"\t\t- {labels.max() + 1} clusters were found and {np.sum(labels == -1)} unclustered contigs")

    # Find the single-copy marker genes on each contig
    print(f"\t- Getting the single-copy marker genes.")
    init_time = time.time()
    contig2markers = markers.get_contig_markers(contigs=contig_seqs, output_dir=output_dir, num_process=-1)
    contig_markers = [contig2markers.get(contig_name, set()) for contig_name in contig_names]
    print(f"\t\t- {sum(len(m) > 0 for m in contig_markers)} contigs have at least one marker gene")
    print(f"\t\t- Done in {time.time() - init_time} seconds.")

    # Select the bins from all clusterings using the marker genes
    print(f"\t- Bin selection started.")
    init_time = time.time()
    bins = bin_selection.select_bins(
        clusterings=clusterings, contig_lens=contig_lens,
        contig_markers=contig_markers, bin_total_min_len=bin_total_min_len
    )
    print(f"\t\t- {len(bins)} bins are selected")
    print(f"\t\t- Done in {time.time() - init_time} seconds.")

    # Save the bins as fasta files and the bin of each contig
    print(f"+ Saving the output bins.")
    init_time = time.time()
    bins_dir = os.path.join(output_dir, "bins")
    os.makedirs(bins_dir, exist_ok=True)
    with open(os.path.join(output_dir, "contig_bins.tsv"), "w") as contig_bins_file:
        contig_bins_file.write("contig_name\tbin\n")
        for bin_id, contig_ids in enumerate(bins):
            with open(os.path.join(bins_dir, f"bin_{bin_id}.fa"), "w") as bin_file:
                for i in contig_ids:
                    bin_file.write(f">{contig_names[i]}\n{contig_seqs[contig_names[i]]}\n")
                    contig_bins_file.write(f"{contig_names[i]}\tbin_{bin_id}\n")
    print(f"\t- Done in {time.time() - init_time} seconds.")