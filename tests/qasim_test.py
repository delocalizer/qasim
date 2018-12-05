import tempfile
import unittest

from os.path import join as path_join
from os.path import dirname

from qasim.qasim import VCF

# test resources are located in the current dir
test1vcf = path_join(dirname(__file__), 'test1.vcf')

class TestQasim(unittest.TestCase):

    def test_VCF_create_empty(self):
        vcf = VCF()
        self.assertEqual(vcf.sample, "SAMPLE")
        self.assertEqual(len(vcf.records), 0)

    def test_VCF_fromfile(self):
        vcf = VCF.fromfile(test1vcf, "sample1")
        self.assertEqual(vcf.sample, "sample1")
        self.assertEqual(len(vcf.records), 4)

    def test_VCF_tuples(self):
        vcf = VCF.fromfile(test1vcf)
        CHROM, POS, REF, ALT, GT = vcf.tuples().__next__()
        self.assertEqual(CHROM, "TEST")
        self.assertEqual(POS, 4)
        self.assertEqual(REF, "A")
        self.assertEqual(ALT, "AGG")
        self.assertEqual(GT, "1|1")

    def test_VCF_write(self):
        vcf = VCF.fromfile(test1vcf)
        with tempfile.TemporaryFile(mode='w+t') as fh:
            vcf.write(fh)
            fh.seek(0)
            lines = fh.readlines()
            self.assertEqual(''.join(lines[:6]), vcf.header)
            self.assertEqual(''.join(lines[6:]),
                '\n'.join(
                    [ '\t'.join([ str(r[c]) for c in vcf.columns ])
                        for r in vcf.records
                    ]) + '\n'
                ) 


if __name__ == '__main__':
    unittest.main()
