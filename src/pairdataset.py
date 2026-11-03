import time
import random
import torch
import numpy as np
from src.utils import set_seed, read_fasta_file
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, file_path, tokenizer, neg_sample_per_pos=1000, max_seq_num=0, input_type='fasta', verbose=True, seed=0):
        """
        PairDataset constructor
        """
        # Set the parameters
        self.__input_features = None # A tensor of shape N x D, storing the k-mer profiles
        self.__indices = None # A tensor of shape L x 2, storing the indices of the positive and negative pairs
        self.__labels = None  # A tensor of shape L, storing the labels of the pairs
        self.__tokenizer = tokenizer
        self.__neg_sample_per_pos = neg_sample_per_pos

        # # Set the seed
        set_seed(seed)

        assert input_type in ['fasta'], "Input type must be fasta!"

        if verbose:
            print(f"+ Reading the input file.")
            print(f"\t- File path: {file_path}")
            init_time = time.time()

        # Get the number of contigs
        num_of_contigs = sum(1 for _ in read_fasta_file(fasta_file_path=file_path))

        assert max_seq_num <= num_of_contigs, \
            f"Maximum sequence number {max_seq_num} must be less than the number of contigs {num_of_contigs}."

        # If the max_seq_num is set, then sample the line numbers to read
        if max_seq_num > 0:
            chosen_contigs = random.sample(range(num_of_contigs), max_seq_num)
            chosen_contigs.sort()
        # Otherwise, read all the lines
        else:
            chosen_contigs = list(range(num_of_contigs))

        # Read the file
        chosen_contig_idx = 0
        left_kmer_feature, right_kmer_feature = [], []
        for contig_id, (contig_name, seq) in enumerate(read_fasta_file(fasta_file_path=file_path)):

            if contig_id == chosen_contigs[chosen_contig_idx]:
                half_seq_len = len(seq) // 2
                left_seq, right_seq = seq[:half_seq_len], seq[half_seq_len:]

                left_kmer_feature.append(self.__tokenizer.encode(left_seq))
                right_kmer_feature.append(self.__tokenizer.encode(right_seq))

                # Increment the chosen_line_idx
                chosen_contig_idx += 1
                # If all the chosen lines have been read, then break
                if chosen_contig_idx >= len(chosen_contigs):
                    break

        # Combine the left and right k-mer profiles.
        # The first half of the profiles are the left ones and the second half are the right ones
        self.__input_features = torch.from_numpy(np.asarray(left_kmer_feature + right_kmer_feature)).to(torch.float)

        if verbose:
            print(f"\t- The input file contains: {num_of_contigs} sequences.")
            if max_seq_num > 0:
                print(f"\t- For training, only {max_seq_num} sequences will be used.")
            else:
                print(f"\t- For training, all sequences will be used.")


        # Sample 'pos_indices' positive pair indices
        pos_indices = torch.vstack((
            torch.arange(self.__input_features.shape[0] // 2, dtype=torch.long),
            torch.arange(self.__input_features.shape[0] // 2, dtype=torch.long) + self.__input_features.shape[0] // 2
        ))
        # Sample 'neg_sample_per_pos' negative pair indices for each positive pair
        temp_weights = torch.ones((self.__input_features.shape[0],), dtype=torch.float)
        neg_indices = torch.vstack((
            torch.multinomial(temp_weights, pos_indices.shape[1] * self.__neg_sample_per_pos, replacement=True),
            torch.multinomial(temp_weights, pos_indices.shape[1] * self.__neg_sample_per_pos, replacement=True),
        ))
        # Concatenate the positive and negative indices
        self.__indices = torch.hstack((pos_indices, neg_indices))
        self.__labels = torch.hstack((
            torch.ones((pos_indices.shape[1],), dtype=torch.float),
            torch.zeros((neg_indices.shape[1],), dtype=torch.float))
        )

        if verbose:
            print(f"\t- For training, {pos_indices.shape[1]} positive pairs were constructed.")
            print(f"\t- For training, {neg_indices.shape[1]} negative pairs were constructed.")
            print(f"\t- Completed in {time.time() - init_time:.2f} seconds.")


    def __len__(self):
        """
        Return the number of pairs
        """
        return self.__indices.shape[1]

    def __getitem__(self, idx):

        return torch.index_select(self.__input_features, dim=0, index=self.__indices[0, idx]), torch.index_select(self.__input_features, dim=0, index=self.__indices[1, idx]), self.__labels[idx]

    def get_input_feature_dim(self):
        return self.__input_features.shape[1]