import unittest

import numpy as np

from src.utils import read_fasta_file
from src.tokenizer import KmerConverter


class TestKMerConverter(unittest.TestCase):

    def test_3mers_num_noncanonical(self):

        kc = KmerConverter(k=3, counting_method="noncanonical")

        self.assertEqual(kc.get_kmers_num(), 64)

    def test_4mers_num_noncanonical(self):

        kc = KmerConverter(k=4, counting_method="noncanonical")

        self.assertEqual(kc.get_kmers_num(), 256)

    def test_4mers_num_canonical(self):

        kc = KmerConverter(k=4, counting_method="canonical")

        self.assertEqual(kc.get_kmers_num(), 136)

    def test_seq2kmer_feature(self):

        feature_gt = np.asarray([
            [0.02253010946513315,0.017729301182858204,0.018529435896570697,0.013096942313996413,0.016550155288966112,
             0.007243325197889242,0.011623009946631298,0.011075549353038542,0.015876357635313487,0.010485976406092496],
            [0.023721881196869278,0.018979245391708824,0.014419353351517852,0.00905016749503345,0.015681155538211918,
             0.0071618152570154334,0.012504894861361339,0.011634686456744743,0.0159161118074584,0.008371404939432503],
            [0.019878228828882977,0.013263122987071024,0.012539081551648821,0.008556853656826699,0.009280895092248902,
             0.00414678309561873,0.009741648732972123,0.00799736709309136,0.010366957245382209,0.004673358685016696],
            [0.015231091030162323,0.006565126191101419,0.00984768863013964,0.006565126191101419,0.0036764712447477834,
             0.002232143771570966,0.005777311205732246,0.005120798717924601,0.008009453664278236,0.0042016812349939]
        ])

        kc = KmerConverter(k=4, counting_method="canonical")

        feature_output = []
        for seq in map(lambda x: x[1], read_fasta_file("dataset/example.fasta")):
            feature_output.append(
                kc.seq2kmer_feature(seq=seq, normalized=True)
            )
        feature_output = np.asarray(feature_output)

        np.testing.assert_allclose(feature_gt,feature_output[:5, :10], rtol=1e-7, atol=1e-7)

if __name__ == '__main__':
    unittest.main()