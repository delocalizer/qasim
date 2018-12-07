import numpy as np
import sys
import tempfile
import unittest

from contextlib import contextmanager
from io import StringIO
from os.path import join as path_join
from os.path import dirname
from xml.etree import ElementTree as ET

from qasim.qasim import EXCEPT_MUT, MSG_SKIP_MUT, MSG_CTOR_SEQ_OR_SIZE, \
    VCF, DipSeq, read_fasta, gen_quals, _t_randqual, reseed

# test resources are located in the current dir
test0fa = path_join(dirname(__file__), 'resources/test0.fa')
test1fa = path_join(dirname(__file__), 'resources/test1.fa')
test0vcf = path_join(dirname(__file__), 'resources/test0.vcf')
test1vcf = path_join(dirname(__file__), 'resources/test1.vcf')
test2vcf = path_join(dirname(__file__), 'resources/test2.vcf')
test3grmvcf = path_join(dirname(__file__), 'resources/test3.1.vcf')
test3somvcf = path_join(dirname(__file__), 'resources/test3.2.vcf')
test4vcf = path_join(dirname(__file__), 'resources/test4.vcf')
testqpxml = path_join(dirname(__file__), 'resources/test.qp.xml')
mut_seq = path_join(dirname(__file__), 'resources/mutagen_result.sequence')
mut_vcf = path_join(dirname(__file__), 'resources/mutagen_result.vcf')


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

    def setUp(self):
        self.test0vcf = VCF.fromfile(test0vcf)

    def test_ctor(self):
        vcf = VCF()
        self.assertEqual(vcf.sample, "SAMPLE")
        self.assertEqual(len(vcf.records), 0)

    def test_tuples(self):
        CHROM, POS, REF, ALT, GT = next(self.test0vcf.tuples())
        self.assertEqual(CHROM, "TEST")
        self.assertEqual(POS, 5)
        self.assertEqual(REF, "C")
        self.assertEqual(ALT, "G")
        self.assertEqual(GT, "0|1")

    def test_write(self):
        with tempfile.TemporaryFile(mode='w+t') as fh:
            self.test0vcf.write(fh)
            fh.seek(0)
            lines = fh.readlines()
            self.assertEqual(''.join(lines[:6]), self.test0vcf.header)
            self.assertEqual(
                ''.join(lines[6:]), '\n'.join(
                    '\t'.join(str(r[c]) for c in self.test0vcf.columns)
                    for r in self.test0vcf.records) + '\n')

    def test_fromfile(self):
        vcf = VCF.fromfile(test0vcf, "sample1")
        self.assertEqual(vcf.sample, "sample1")
        self.assertEqual(len(vcf.records), 3)


class TestDipSeq(unittest.TestCase):

    def test_ctor_from_size(self):
        d = DipSeq("T", "TEST", size=6)
        self.assertEqual(d.stopA, 6)
        self.assertEqual(d.stopB, 6)

    def test_ctor_from_seq(self):
        d = DipSeq("T", "TEST", hapseq=bytearray([65, 67, 71, 84, 78, 45]))
        self.assertEqual(d.stopA, 6)
        self.assertEqual(d.stopB, 6)
        self.assertEqual(list(d.seqA), [0, 1, 2, 3, 4, 5])

    def test_ctor_seq_and_size(self):
        with self.assertRaisesRegex(Exception, MSG_CTOR_SEQ_OR_SIZE):
            DipSeq("T", "TEST", bytearray([65]), 1)

    def test_write(self):
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
        # data in the test resource files are valid "by inspection" and
        # require that the test run matches them every time. In this case
        # cross-referencing the randomly generated mutations in the vcf with
        # the transformed sequence file is what's required. You should take
        # a look — it's interesting!
        refseq = next(read_fasta(test1fa))
        vcf = VCF("sample1")
        mut_rate = 0.01
        homo_frac = 0.333333
        indel_frac = 0.15
        indel_extend = 0.3
        max_insertion = 1000
        reseed(12345678)  # deterministic iff we set the seed
        DipSeq.mutagen(refseq, vcf, mut_rate, homo_frac, indel_frac,
                       indel_extend, max_insertion)
        out = StringIO()
        vcf.write(out)
        with open(mut_vcf) as fh:
            self.assertEqual(out.getvalue(), ''.join(fh.readlines()))
        mutseq = DipSeq(
            refseq.seqid + '.mut',
            refseq.description,
            size=refseq.seqA.shape[0] * 2,
            fold=refseq.fold)
        mutseq.transform(refseq, vcf)
        out = StringIO()
        mutseq.write(out)
        with open(mut_seq) as fh:
            self.assertEqual(out.getvalue(), ''.join(fh.readlines()))

    def test_transform_0(self):
        """germline het & hom snps"""
        vcf = VCF.fromfile(test0vcf, "sample1")
        refseq = next(read_fasta(test0fa))
        mutseq = DipSeq(
            refseq.seqid + '.mut',
            refseq.description,
            size=refseq.seqA.shape[0] * 2,
            fold=refseq.fold)
        mutseq.transform(refseq, vcf)
        for i in range(refseq.stopA):
            POS = i + 1               # VCF coords
            if (POS == 5):
                self.assertEqual(base(mutseq.seqA[i]), base(refseq.seqA[i]))
                self.assertEqual(base(mutseq.seqB[i]), "G")
            elif (POS == 9):
                self.assertEqual(base(mutseq.seqA[i]), "G")
                self.assertEqual(base(mutseq.seqB[i]), base(refseq.seqA[i]))
            elif (POS == 13):
                self.assertEqual(base(mutseq.seqA[i]), "T")
                self.assertEqual(base(mutseq.seqA[i]), base(mutseq.seqB[i]))
                self.assertNotEqual(base(mutseq.seqA[i]), base(refseq.seqA[i]))
            else:
                self.assertEqual(base(mutseq.seqA[i]), base(refseq.seqA[i]))
                self.assertEqual(base(mutseq.seqB[i]), base(refseq.seqA[i]))

    def test_transform_1(self):
        """simple het & hom indels"""
        vcf = VCF.fromfile(test1vcf, "sample1")
        refseq = next(read_fasta(test0fa))
        mutseq = DipSeq(
            refseq.seqid + '.mut',
            refseq.description,
            size=refseq.seqA.shape[0] * 2,
            fold=refseq.fold)
        mutseq.transform(refseq, vcf)
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
        vcf = VCF.fromfile(test2vcf, "sample1")
        refseq = next(read_fasta(test0fa))
        mutseq = DipSeq(
            refseq.seqid + '.mut',
            refseq.description,
            size=refseq.seqA.shape[0] * 2,
            fold=refseq.fold)
        mutseq.transform(refseq, vcf)
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
        grmvcf = VCF.fromfile(test3grmvcf)
        somvcf = VCF.fromfile(test3somvcf)
        refseq = next(read_fasta(test0fa))

        grmseq = DipSeq(
            refseq.seqid + '.grm',
            refseq.description,
            size=refseq.seqA.shape[0] * 2,
            fold=refseq.fold)
        grmseq.transform(refseq, grmvcf)
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
            somseq.transform(grmseq, somvcf)
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
        vcf = VCF.fromfile(test4vcf, "sample1")
        refseq = next(read_fasta(test0fa))
        mutseq = DipSeq(
            refseq.seqid + '.mut',
            refseq.description,
            size=refseq.seqA.shape[0] * 2,
            fold=refseq.fold)
        expected_msg = EXCEPT_MUT % {'POS': 7, 'OLDPOS': 5}
        with self.assertRaisesRegex(Exception, expected_msg):
            mutseq.transform(refseq, vcf)


class TestQasim(unittest.TestCase):

    def test_read_fasta(self):
        for seq in read_fasta(test0fa):
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
        mean_1 = sum(_t_randqual(dist, 1) for i in range(N)) / float(N)
        mean_2 = sum(_t_randqual(dist, 2) for i in range(N)) / float(N)
        mean_3 = sum(_t_randqual(dist, 3) for i in range(N)) / float(N)
        mean_4 = sum(_t_randqual(dist, 4) for i in range(N)) / float(N)
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
        gen_quals(testqpxml, read_length, num_quals, qvals, pvals)

        for sample in range(num_quals):
            for q, p in zip(qvals[sample], pvals[sample]):
                self.assertEqual(p, 10 ** (q / -10))

        doc = ET.parse(testqpxml)
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


if __name__ == '__main__':
    unittest.main()
