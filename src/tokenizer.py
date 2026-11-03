import itertools
import numpy as np
from src.utils import read_fasta_file


_LETTERS = ['A', 'T', 'G', 'C']
_BASE_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

class KmerTokenizer:
    def __init__(self, k, counting_method = "canonical"):

        # Define the letters, k-mer size, and the base complement
        self.__k = k
        # Define the k-mers
        self.__kmer2id = dict()
        kmers_num = 0
        for kmer in itertools.product(_LETTERS, repeat=self.__k):
            kmer = "".join(kmer)
            if kmer not in self.__kmer2id:
                self.__kmer2id[kmer] = kmers_num

                if counting_method == "canonical":
                    # If kmer is not in the list, then its k-mer isn't as well.
                    kmer_reverse_complement = "".join(map(_BASE_COMPLEMENT.get, reversed(kmer)))
                    self.__kmer2id[kmer_reverse_complement] = kmers_num

                # Increment the k-mers count by 1
                kmers_num += 1

        # Set the k-mer count
        self.__kmers_num = kmers_num

    def get_k(self):
        return self.__k

    def get_kmers_num(self):
        return self.__kmers_num

    def encode(self, seq:str, normalized:bool=True):

        # TODO: Decide how to handle the non-standard nucleotides
        kmer2id = [
            self.__kmer2id[seq[i:i + self.__k]] for i in range(len(seq) - self.__k + 1)
            if seq[i:i + self.__k] in self.__kmer2id
        ]

        kmer_feature = np.bincount(kmer2id, minlength=self.__kmers_num)

        if normalized:
            kmer_feature = kmer_feature / kmer_feature.sum()

        return kmer_feature

    # def seq2kmer_feature(self, seq:list, normalized:bool=True):
    #
    #     # TODO: Decide how to handle the non-standard nucleotides
    #     kmer2id = [
    #         self.__kmer2id[seq[i:i + self.__k]] for i in range(len(seq) - self.__k + 1)
    #         if seq[i:i + self.__k] in self.__kmer2id
    #     ]
    #
    #     kmer_profile = np.bincount(kmer2id, minlength=self.__kmers_num)
    #
    #     if normalized:
    #         kmer_profile = kmer_profile / kmer_profile.sum()
    #
    #     return kmer_profile
    #
    # def fasta_file_to_kmer_feature_pairs(self, fasta_file, normalized:bool=True):
    #
    #     identifiers, features = [], []
    #
    #     for seq_identifier, seq in read_fasta_file(fasta_file):
    #
    #         half_len = len(seq) // 2
    #         seq_left, seq_right = seq[:half_len], seq[half_len:]
    #
    #         left_features = self.seq2kmer_feature(seq_left, normalized=normalized)
    #         right_features = self.seq2kmer_feature(seq_right, normalized=normalized)
    #
    #         identifiers.append(seq_identifier + "_left")
    #         features.append(left_features)
    #         identifiers.append(seq_identifier + "_right")
    #         features.append(right_features)
    #
    #     return identifiers, features