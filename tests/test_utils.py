import unittest

class TestReadFile(unittest.TestCase):

    def test_read_file(self):

        from src.utils import read_fasta_file
        couunter = 0
        for out in read_fasta_file("dataset/example.fasta"):
            print(couunter, out)
            couunter += 1
            # self.assertEqual(3,3)