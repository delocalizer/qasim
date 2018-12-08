"""Tests for qasim package"""
import sys
import tempfile
import unittest

from contextlib import contextmanager
from io import StringIO
from os.path import join as path_join
from os.path import dirname
from xml.etree import ElementTree as ET

import numpy as np

from qasim import qasim
from qasim.qasim import EXCEPT_MUT, MSG_SKIP_MUT, MSG_CTOR_SEQ_OR_SIZE
from qasim.qasim import VCF, DipSeq

@contextmanager
def captured_output():
    """
    https://stackoverflow.com/a/17981937/6705037
    """
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def base(code):
    """
    Return letter code for integer encoded base
    """
    return "ACGTN-"[code]


class TestVcf(unittest.TestCase):
    """VCF class tests"""

    @classmethod
    def setUpClass(cls):
        """Load some test resources"""
        # Notify aggressively so we know where we are if setup fails
        sys.stdout.write("\nTestVcf.setUpClass\n")
        sys.stdout.flush()
        cls.vcf0 = VCF.fromfile(
            path_join(dirname(__file__), 'resources/test.vcf.0'))

    def test_ctor(self):
        """Test the no-args constructor"""
        vcf = VCF()
        self.assertEqual(vcf.sample, "SAMPLE")
        self.assertEqual(len(vcf.records), 0)

    def test_tuples(self):
        """Test the VCF.tuples() method"""
        CHROM, POS, REF, ALT, GT = next(self.vcf0.tuples())
        self.assertEqual(CHROM, "TEST")
        self.assertEqual(POS, 5)
        self.assertEqual(REF, "C")
        self.assertEqual(ALT, "G")
        self.assertEqual(GT, "0|1")

    def test_write(self):
        """Test the VCF.write() method"""
        with tempfile.TemporaryFile(mode='w+t') as fh:
            self.vcf0.write(fh)
            fh.seek(0)
            lines = fh.readlines()
            self.assertEqual(''.join(lines[:6]), self.vcf0.header)
            self.assertEqual(
                ''.join(lines[6:]), '\n'.join(
                    '\t'.join(str(r[c]) for c in self.vcf0.columns)
                    for r in self.vcf0.records) + '\n')

    def test_fromfile(self):
        """Test the VCF.fromfile() method"""
        vcf = VCF.fromfile(
            path_join(dirname(__file__), 'resources/test.vcf.0'), "sample1")
        self.assertEqual(vcf.sample, "sample1")
        self.assertEqual(len(vcf.records), 3)


class TestDipSeq(unittest.TestCase):
    """DipSeq class tests"""

    @classmethod
    def setUpClass(cls):
        """Load some test resources"""
        # Notify aggressively so we know where we are if setup fails
        sys.stdout.write("\nTestDipSeq.setUpClass\n")
        sys.stdout.flush()
        cls.mut_seq = path_join(
            dirname(__file__), 'resources/mutagen_result.sequence')
        cls.mut_vcf = path_join(
            dirname(__file__), 'resources/mutagen_result.vcf')
        cls.fa0 = next(qasim.read_fasta(
            path_join(dirname(__file__), 'resources/test.fa.0')))
        cls.fa1 = next(qasim.read_fasta(
            path_join(dirname(__file__), 'resources/test.fa.1')))
        cls.vcf0 = VCF.fromfile(
            path_join(dirname(__file__), 'resources/test.vcf.0'))
        cls.vcf1 = VCF.fromfile(
            path_join(dirname(__file__), 'resources/test.vcf.1'))
        cls.vcf2 = VCF.fromfile(
            path_join(dirname(__file__), 'resources/test.vcf.2'))
        cls.vcfgrm = VCF.fromfile(
            path_join(dirname(__file__), 'resources/test.vcf.3_1'))
        cls.vcfsom = VCF.fromfile(
            path_join(dirname(__file__), 'resources/test.vcf.3_2'))
        cls.vcf4 = VCF.fromfile(
            path_join(dirname(__file__), 'resources/test.vcf.4'))

    def test_ctor_from_size(self):
        """Test the constructor that takes size argument"""
        d = DipSeq("T", "TEST", size=6)
        self.assertEqual(d.stopA, 6)
        self.assertEqual(d.stopB, 6)

    def test_ctor_from_seq(self):
        """Test the constructor that takes a sequence argument"""
        d = DipSeq("T", "TEST", hapseq=bytearray([65, 67, 71, 84, 78, 45]))
        self.assertEqual(d.stopA, 6)
        self.assertEqual(d.stopB, 6)
        self.assertEqual(list(d.seqA), [0, 1, 2, 3, 4, 5])

    def test_ctor_seq_and_size(self):
        """Test Exception is correctly raised for bad ctor args"""
        with self.assertRaisesRegex(Exception, MSG_CTOR_SEQ_OR_SIZE):
            DipSeq("T", "TEST", bytearray([65]), 1)

    def test_write(self):
        """Test the DipSeq.write() method"""
        d = DipSeq("T", "TEST", hapseq=bytearray([65, 67, 71, 84, 78]))
        out = StringIO()
        d.write(out)
        self.assertEqual(out.getvalue(), (
            ">T.0 TEST\n"
            "ACGTN\n"
            "12345\n"
            ">T.1 TEST\n"
            "ACGTN\n"
            "12345\n"))

    def test_mutagen(self):
        """tests both mutagen and transform methods"""
        # Another case where we take a shortcut by simply asserting that the
        # data in the test resource files mut_vcf and mut_seq are valid "by
        # inspection" and require that the test run matches them every time.
        # In this case cross-referencing the randomly generated mutations in
        # the vcf with the transformed sequence file is what's required. You
        # should eyeball it yourself — it's interesting!
        refseq = self.fa1
        vcf = VCF("sample1")
        mut_rate = 0.01
        homo_frac = 0.333333
        indel_frac = 0.15
        indel_extend = 0.3
        max_insertion = 1000
        qasim.reseed(12345678)  # deterministic iff we set the seed
        DipSeq.mutagen(refseq, vcf, mut_rate, homo_frac, indel_frac,
                       indel_extend, max_insertion)
        out = StringIO()
        vcf.write(out)
        with open(self.mut_vcf) as fh:
            self.assertEqual(out.getvalue(), ''.join(fh.readlines()))
        mutseq = DipSeq(
            refseq.seqid + '.mut',
            refseq.description,
            size=refseq.seqA.shape[0] * 2,
            fold=refseq.fold)
        mutseq.transform(refseq, vcf)
        out = StringIO()
        mutseq.write(out)
        with open(self.mut_seq) as fh:
            self.assertEqual(out.getvalue(), ''.join(fh.readlines()))

    def test_transform_0(self):
        """germline het & hom snps"""
        refseq = self.fa0
        mutseq = DipSeq(
            refseq.seqid + '.mut',
            refseq.description,
            size=refseq.seqA.shape[0] * 2,
            fold=refseq.fold)
        mutseq.transform(refseq, self.vcf0)
        for i in range(refseq.stopA):
            POS = i + 1               # VCF coords
            if POS == 5:
                self.assertEqual(base(mutseq.seqA[i]), base(refseq.seqA[i]))
                self.assertEqual(base(mutseq.seqB[i]), "G")
            elif POS == 9:
                self.assertEqual(base(mutseq.seqA[i]), "G")
                self.assertEqual(base(mutseq.seqB[i]), base(refseq.seqA[i]))
            elif POS == 13:
                self.assertEqual(base(mutseq.seqA[i]), "T")
                self.assertEqual(base(mutseq.seqA[i]), base(mutseq.seqB[i]))
                self.assertNotEqual(base(mutseq.seqA[i]), base(refseq.seqA[i]))
            else:
                self.assertEqual(base(mutseq.seqA[i]), base(refseq.seqA[i]))
                self.assertEqual(base(mutseq.seqB[i]), base(refseq.seqA[i]))

    def test_transform_1(self):
        """simple het & hom indels"""
        refseq = self.fa0
        mutseq = DipSeq(
            refseq.seqid + '.mut',
            refseq.description,
            size=refseq.seqA.shape[0] * 2,
            fold=refseq.fold)
        mutseq.transform(refseq, self.vcf1)
        insertion_1 = range(4, 7)
        self.assertEqual(
            ''.join(base(mutseq.seqA[POS - 1]) for POS in insertion_1), "AGG")
        self.assertEqual(
            list(mutseq.relA[POS - 1] for POS in insertion_1), [4, 4, 4])
        insertion_2 = range(5, 9)
        self.assertEqual(
            ''.join(base(mutseq.seqB[POS - 1]) for POS in insertion_2), "CTTT")
        self.assertEqual(
            list(mutseq.relB[POS - 1] for POS in insertion_2), [5, 5, 5, 5])
        deletion_1A = range(15, 17)
        deletion_1B = range(16, 18)
        self.assertEqual(
            ''.join(base(mutseq.seqA[POS - 1]) for POS in deletion_1A), "CC")
        self.assertEqual(
            list(mutseq.relA[POS - 1] for POS in deletion_1A), [13, 16])
        self.assertEqual(
            ''.join(base(mutseq.seqB[POS - 1]) for POS in deletion_1B), "CC")
        self.assertEqual(
            list(mutseq.relB[POS - 1] for POS in deletion_1B), [13, 16])

    def test_transform_2(self):
        """complex overlapping mutations"""
        refseq = self.fa0
        mutseq = DipSeq(
            refseq.seqid + '.mut',
            refseq.description,
            size=refseq.seqA.shape[0] * 2,
            fold=refseq.fold)
        mutseq.transform(refseq, self.vcf2)
        out = StringIO()
        mutseq.write(out)
        # we take a bit of a shorcut and rather than testing all the
        # explicit logic of the transformation we just test equality to
        # this output that we assert is valid "by inspection".
        self.assertEqual(out.getvalue(), (
            ">TEST.mut.0 small fasta for testing\n"
            "AAAAGGCCGAAACCCC\n"
            "1234445690123456\n"
            ">TEST.mut.1 small fasta for testing\n"
            "AAAAGGCTTTCAAAACCCC\n"
            "1234445555690123456\n"))

    def test_transform_3(self):
        """overlapping mutations in somatic mode"""
        refseq = self.fa0

        grmseq = DipSeq(
            refseq.seqid + '.grm',
            refseq.description,
            size=refseq.seqA.shape[0] * 2,
            fold=refseq.fold)
        grmseq.transform(refseq, self.vcfgrm)
        out = StringIO()
        grmseq.write(out)
        self.assertEqual(out.getvalue(), (
            ">TEST.grm.0 small fasta for testing\n"
            "AAAAGGCCCCAAAACCCC\n"
            "123444567890123456\n"
            ">TEST.grm.1 small fasta for testing\n"
            "AAAAGGCCAAAACCCC\n"
            "1234447890123456\n"))

        somseq = DipSeq(
            refseq.seqid + '.som',
            refseq.description,
            size=refseq.seqA.shape[0] * 2,
            fold=refseq.fold)
        with captured_output() as (out, err):
            expected_msg = MSG_SKIP_MUT % {'allele': 1, 'POS': 5}
            somseq.transform(grmseq, self.vcfsom)
            self.assertEqual(err.getvalue(), expected_msg)
        out = StringIO()
        somseq.write(out)
        # somatic insertion at 4 and deletion at 5 both applied to allele 0
        # somatic deletion at 5 isn't applied to allele 1
        self.assertEqual(out.getvalue(), (
            ">TEST.som.0 small fasta for testing\n"
            "AAAATTGGCAAAACCCC\n"
            "12344444590123456\n"
            ">TEST.som.1 small fasta for testing\n"
            "AAAATTGGCCAAAACCCC\n"
            "123444447890123456\n"))

    def test_transform_4(self):
        """disallow mutations overlapping deletions in same vcf"""
        refseq = self.fa0
        mutseq = DipSeq(
            refseq.seqid + '.mut',
            refseq.description,
            size=refseq.seqA.shape[0] * 2,
            fold=refseq.fold)
        expected_msg = EXCEPT_MUT % {'POS': 7, 'OLDPOS': 5}
        with self.assertRaisesRegex(Exception, expected_msg):
            mutseq.transform(refseq, self.vcf4)


class TestQasim(unittest.TestCase):
    """Test module-level functions"""

    @classmethod
    def setUpClass(cls):
        """Load some test resources"""
        # Notify aggressively so we know where we are if setup fails
        sys.stdout.write("\nTestQasim.setUpClass\n")
        sys.stdout.flush()
        cls.fa1 = path_join(dirname(__file__), 'resources/test.fa.1')
        cls.vcfgrm = path_join(dirname(__file__), 'resources/germline.vcf')
        cls.vcfsom = path_join(dirname(__file__), 'resources/somatic.vcf')
        cls.qpxml = path_join(dirname(__file__), 'resources/test.qp.xml')

    def test_read_fasta(self):
        """Test read_fasta()"""
        for seq in qasim.read_fasta(
                path_join(dirname(__file__), 'resources/test.fa.0')):
            out = StringIO()
            seq.write(out)
            self.assertEqual(out.getvalue(), (
                ">TEST.0 small fasta for testing\n"
                "AAAACCCCAAAACCCC\n"
                "1234567890123456\n"
                ">TEST.1 small fasta for testing\n"
                "AAAACCCCAAAACCCC\n"
                "1234567890123456\n"))

    def test_randqual(self):
        """random choice from cumulative frequency distribution"""
        # dist = { cycle: (qual, cumulative_count) }
        dist = {
            # All 8 quals equally likely
            1: [(2, 1000), (8, 2000), (12, 3000), (22, 4000), (27, 5000),
                (32, 6000), (37, 7000), (41, 8000)],
            # high-end bias: 41 and 37 equally likely, others zero
            2: [(2, 0), (8, 0), (12, 0), (22, 0), (27, 0), (32, 0), (37, 4000),
                (41, 8000)],
            # very high-end bias: 41 is 7x more likely than 37, others zero
            3: [(2, 0), (8, 0), (12, 0), (22, 0), (27, 0), (32, 0), (37, 1000),
                (41, 8000)],
            # low-end bias: 2 is 7x more likely than 8, others zero
            4: [(2, 7000), (8, 8000), (12, 8000), (22, 8000), (27, 8000),
                (32, 8000), (37, 8000), (41, 8000)]
        }

        def avg(counts):
            """counts = [(qual, cumulative_count), ...]"""
            return sum(
                counts[i][0] *
                (counts[i][1] - (0 if i == 0 else counts[i - 1][1]))
                for i in range(len(counts))) / float(counts[-1][1])

        mu_1 = avg(dist[1])  # = 22.625
        mu_2 = avg(dist[2])  # = 39
        mu_3 = avg(dist[3])  # = 40.5
        mu_4 = avg(dist[4])  # = 2.75
        N = 100000
        mean_1 = sum(qasim._t_randqual(dist, 1) for i in range(N)) / float(N)
        mean_2 = sum(qasim._t_randqual(dist, 2) for i in range(N)) / float(N)
        mean_3 = sum(qasim._t_randqual(dist, 3) for i in range(N)) / float(N)
        mean_4 = sum(qasim._t_randqual(dist, 4) for i in range(N)) / float(N)
        self.assertAlmostEqual(mean_1 / mu_1, 1.0, delta=0.01)
        self.assertAlmostEqual(mean_2 / mu_2, 1.0, delta=0.01)
        self.assertAlmostEqual(mean_3 / mu_3, 1.0, delta=0.01)
        self.assertAlmostEqual(mean_4 / mu_4, 1.0, delta=0.01)

    def test_gen_quals(self):
        """check that P values match Q scores, and sample is representative"""
        read_length = 150
        num_quals = 10000
        qvals = np.ndarray((num_quals, read_length), dtype='u1')
        pvals = np.ndarray((num_quals, read_length))
        qasim.gen_quals(self.qpxml, read_length, num_quals, qvals, pvals)

        for sample in range(num_quals):
            for q, p in zip(qvals[sample], pvals[sample]):
                # Phred definition
                self.assertEqual(p, 10 ** (q / -10))

        doc = ET.parse(self.qpxml)
        Q = doc.getroot().find('.//QUAL')
        for cyclenum in range(1, read_length + 1):
            cycle = Q.find(".//Cycle[@value='%s']" % cyclenum)
            weights = sum(int(t.get('value')) * int(t.get('count'))
                          for t in cycle.findall('TallyItem'))
            counts = sum(int(t.get('count'))
                         for t in cycle.findall('TallyItem'))
            mu_qual = weights / float(counts)               # population mean
            samples = qvals[:, cyclenum - 1]
            mean_qual = sum(samples) / float(len(samples))  # sample mean
            self.assertAlmostEqual(mean_qual / mu_qual, 1.0, delta=0.01)

    def test_integration_1(self):
        """somatic mode with mutations specified by input VCFs"""
        grm_out1 = path_join(dirname(__file__), "control.30x.read1.fastq")
        grm_out2 = path_join(dirname(__file__), "control.30x.read2.fastq")
        grm_args = [
            "--seed", "12345678",
            "--sample-name", "c9a6be94-bdb7-4c0d-a89d-4addbf76e486",
            "--vcf-input", self.vcfgrm,
            "--num-pairs", "160",
            "--quals-from", self.qpxml,
            "--length1", "150",
            "--length2", "150",
            self.fa1,
            grm_out1,
            grm_out2]
        qasim.workflow(qasim.get_args(grm_args))

        som_out1 = path_join(dirname(__file__), "test.60x.read1.fastq")
        som_out2 = path_join(dirname(__file__), "test.60x.read2.fastq")
        som_args = [
            "--seed", "12345678",
            "--sample-name", "c9a6be94-bdb7-4c0d-a89d-4addbf76e486",
            "--vcf-input", self.vcfgrm,
            "--somatic-mode",
            "--sample-name2", "d44d739c-0143-4350-bba5-72dd068e05fd",
            "--contamination", "0.3",
            "--vcf-input2", self.vcfsom,
            "--num-pairs", "320",
            "--quals-from", self.qpxml,
            "--length1", "150",
            "--length2", "150",
            self.fa1,
            som_out1,
            som_out2]
        qasim.workflow(qasim.get_args(som_args))


if __name__ == '__main__':
    unittest.main()
