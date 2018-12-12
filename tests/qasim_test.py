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
from .testutil import Fastq


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


class TestTestUtils(unittest.TestCase):
    """Test the testing framework's methods"""

    @classmethod
    def setUpClass(cls):
        """Load some test resources"""
        # Notify aggressively so we know where we are if setup fails
        sys.stdout.write("\nTestTestUtils.setUpClass\n")
        sys.stdout.flush()
        cls.fq1 = path_join(dirname(__file__), 'resources/test.30x.1.fq')
        cls.fq2 = path_join(dirname(__file__), 'resources/test.30x.2.fq')

    def test_Fastq(self):
        """Test the Fastq class"""
        fq1 = Fastq(self.fq1)
        self.assertEqual(len(fq1.records), 160)
        self.assertEqual(fq1.read_length, 150)
        self.assertEqual(fq1.records[-1]['id'], "@TEST.1_965_1528_e0_e2_9f/1")

    def test_forwardization_and_coverage(self):
        """Confirm that coverages calculated two ways agree"""
        fq1 = Fastq(self.fq1)
        fq2 = Fastq(self.fq2)
        fwd1 = Fastq.forwardize(fq1)
        fwd2 = Fastq.forwardize(fq2)
        low = min(fq1.minpos, fq2.minpos)
        high = max(fq1.maxpos, fq2.maxpos)
        for pos in range(low, high + 1):
            cov1 = set(r['id'] for r in fq1.coverage(pos) + fq2.coverage(pos))
            cov2 = set(r['id'] for r in fwd1.coverage(pos) + fwd2.coverage(pos))
            self.assertEqual(cov1, cov2)


class TestQasim(unittest.TestCase):
    """Test module-level functions"""

    @classmethod
    def setUpClass(cls):
        """Load some test resources"""
        # Notify aggressively so we know where we are if setup fails
        sys.stdout.write("\nTestQasim.setUpClass\n")
        sys.stdout.flush()
        cls.fa0 = path_join(dirname(__file__), 'resources/test.fa.0')
        cls.fa1 = path_join(dirname(__file__), 'resources/test.fa.1')
        cls.fa2 = path_join(dirname(__file__), 'resources/test.fa.2')
        cls.vcfgrm = path_join(dirname(__file__), 'resources/germline.vcf')
        cls.vcfsom = path_join(dirname(__file__), 'resources/somatic.vcf')
        cls.vcfindel = path_join(dirname(__file__), 'resources/indel.vcf.1')
        cls.qpxml = path_join(dirname(__file__), 'resources/test.qp.xml')
        cls.fq1 = path_join(dirname(__file__), 'resources/test.60x.1.fq')
        cls.fq2 = path_join(dirname(__file__), 'resources/test.60x.2.fq')

    def test_read_fasta(self):
        """Test read_fasta()"""
        seq = next(qasim.read_fasta(self.fa0))
        with StringIO() as out:
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

    def test_integration_0(self):
        """check for any & all changes to output"""
        out1 = path_join(dirname(__file__), "test_integration_0.1.fq")
        out2 = path_join(dirname(__file__), "test_integration_0.2.fq")
        read_length = 150
        contamination = 0.3
        args = [
            "--seed", "12345678",
            "--sample-name", "c9a6be94-bdb7-4c0d-a89d-4addbf76e486",
            "--vcf-input", self.vcfgrm,
            "--somatic-mode",
            "--sample-name2", "d44d739c-0143-4350-bba5-72dd068e05fd",
            "--contamination", str(contamination),
            "--vcf-input2", self.vcfsom,
            "--num-pairs", "320",
            "--quals-from", self.qpxml,
            "--length1", str(read_length),
            "--length2", str(read_length),
            self.fa1,
            out1,
            out2]
        qasim.workflow(qasim.get_args(args))
        # There's no complicated logic here, just run a deterministic workflow
        # and compare the output to what it was when we wrote the tests.
        with open(out1) as test, open(self.fq1) as original:
            test_content = test.readlines()
            original_content = original.readlines()
            self.assertEqual(test_content, original_content)
        with open(out2) as test, open(self.fq2) as original:
            test_content = test.readlines()
            original_content = original.readlines()
            self.assertEqual(test_content, original_content)

    def test_integration_1(self):
        """check reads are generated correctly over indels"""
        out1 = path_join(dirname(__file__), "test_integration_1.1.fq")
        out2 = path_join(dirname(__file__), "test_integration_1.2.fq")
        read_length = 150
        # generate reads with no sequencing errors to make comparison back
        # to reference easy:
        args = [
            "--seed", "12345678",
            "--sample-name", "c9a6be94-bdb7-4c0d-a89d-4addbf76e486",
            "--vcf-input", self.vcfindel,
            "--num-pairs", "320",
            "--error-rate", "0",
            "--length1", str(read_length),
            "--length2", str(read_length),
            self.fa2,
            out1,
            out2]
        qasim.workflow(qasim.get_args(args))
        ref = next(qasim.read_fasta(self.fa2))
        # pad with ' ' to make 1-based reference sequence as a string
        ref_seq = ' ' + ''.join(base(b) for b in ref.seqA)
        self.assertEqual(ref_seq[1:33], "AAAAAAAACCCCCCCCGGGGGGGGTTTTTTTT")
        fq1 = Fastq(out1)
        fq2 = Fastq(out2)
        # The VCF specifies a homozygous insertion A>ACG at POS=401:
        pos = 401
        ins_reads = fq1.coverage(pos) + fq2.coverage(pos)

        # Check the forward reads over the insertion
        fwd_reads = [r for r in ins_reads if
                     r['read'] == 1 and r['frag_start'] < r['frag_end'] or
                     r['read'] == 2 and r['frag_start'] > r['frag_end']]
        for r in fwd_reads:
            read_start = min(r['frag_start'], r['frag_end'])
            for i, b in enumerate(r['seq']):
                b_pos = read_start + i
                if b_pos <= pos:
                    self.assertEqual(b, ref_seq[b_pos])
                elif b_pos == pos + 1:
                    self.assertEqual(b, 'C')
                elif b_pos == pos + 2:
                    self.assertEqual(b, 'G')
                else:
                    self.assertEqual(b, ref_seq[b_pos - 2])

        # Check the reverse reads over the insertion
        rev_reads = [r for r in ins_reads if
                     r['read'] == 1 and r['frag_start'] > r['frag_end'] or
                     r['read'] == 2 and r['frag_start'] < r['frag_end']]
        for r in rev_reads:
            read_start = max(r['frag_start'], r['frag_end'])
            if read_start == pos :
                # skip reverse reads whose start coord (far end) is /exactly/
                # pos since the indel isn't actually contained in these reads
                continue
            for i, b in enumerate(r['seq']):
                # b_pos is decreasing as we read backwards
                b_pos = read_start - i
                b = Fastq.complement[b]
                if b_pos > pos:
                    self.assertEqual(b, ref_seq[b_pos])
                elif b_pos == pos:
                    self.assertEqual(b, 'G')
                elif b_pos == pos - 1:
                    self.assertEqual(b, 'C')
                else:
                    self.assertEqual(b, ref_seq[b_pos + 2])

        # The VCF specifies a homozygous deletion AAA>A at POS=1201.
        pos = 1201
        del_reads = fq1.coverage(pos) + fq2.coverage(pos)

        # Check the forward reads over the deletion
        fwd_reads = [r for r in del_reads if
                     r['read'] == 1 and r['frag_start'] < r['frag_end'] or
                     r['read'] == 2 and r['frag_start'] > r['frag_end']]
        for r in fwd_reads:
            read_start = min(r['frag_start'], r['frag_end'])
            for i, b in enumerate(r['seq']):
                b_pos = read_start + i
                if b_pos <= pos:
                    self.assertEqual(b, ref_seq[b_pos])
                else:
                    self.assertEqual(b, ref_seq[b_pos + 2])

        # Check the reverse reads over the deletion
        rev_reads = [r for r in del_reads if
                     r['read'] == 1 and r['frag_start'] > r['frag_end'] or
                     r['read'] == 2 and r['frag_start'] < r['frag_end']]
        for r in rev_reads:
            read_start = max(r['frag_start'], r['frag_end'])
            if read_start == pos :
                # skip reverse reads whose start coord (far end) is /exactly/
                # pos since the indel isn't actually contained in these reads
                continue
            for i, b in enumerate(r['seq']):
                # b_pos is decreasing as we read backwards
                b_pos = read_start - i
                b = Fastq.complement[b]
                if b_pos >= pos + 2:
                    self.assertEqual(b, ref_seq[b_pos])
                else:
                    self.assertEqual(b, ref_seq[b_pos - 2])

    def test_integration_2(self):
        """germline mode with mutations specified by input VCF"""
        # We take advantage of the fact that we know the true location
        # of the generated reads on the reference (from the coord1_coord2
        # embedded in read ids) to check SNP genotypes at positions without
        # having to align the reads first. This wouldn't be straightforward
        # for indels because the insertion/deletion shifts the coordinates.
        out1 = path_join(dirname(__file__), "test_integration_2.1.fq")
        out2 = path_join(dirname(__file__), "test_integration_2.2.fq")
        read_length = 150
        args = [
            "--seed", "12345678",
            "--sample-name", "c9a6be94-bdb7-4c0d-a89d-4addbf76e486",
            "--vcf-input", self.vcfgrm,
            "--num-pairs", "160",
            "--quals-from", self.qpxml,
            "--length1", str(read_length),
            "--length2", str(read_length),
            self.fa1,
            out1,
            out2]
        qasim.workflow(qasim.get_args(args))
        # work with "forwardized" reads: it's more convenient to only deal
        # with variants relative to the reference strand.
        fq1 = Fastq.forwardize(Fastq(out1))
        fq2 = Fastq.forwardize(Fastq(out2))
        # In the assertions below we're quite lenient to account for both
        # sequencing errors (introducing non-REF/ALT bases) and in the case
        # of the first two het positions, imbalanced read coverage of the
        # A & B alleles. The variants here are specified in `self.vcfgrm`
        # A>C 0|1 SNP at position 81
        pos = 81
        covering_reads = fq1.coverage(pos) + fq2.coverage(pos)
        pos_bases = [r['seq'][pos - r['read_start']] for r in covering_reads]
        frac_A = pos_bases.count('A')/float(len(pos_bases))
        frac_C = pos_bases.count('C')/float(len(pos_bases))
        self.assertAlmostEqual(frac_A, 0.5, delta=0.1)
        self.assertAlmostEqual(frac_C, 0.5, delta=0.1)
        # A>C 1|0 SNP at position 161
        pos = 161
        covering_reads = fq1.coverage(pos) + fq2.coverage(pos)
        pos_bases = [r['seq'][pos - r['read_start']] for r in covering_reads]
        frac_A = pos_bases.count('A')/float(len(pos_bases))
        frac_C = pos_bases.count('C')/float(len(pos_bases))
        self.assertAlmostEqual(frac_A, 0.5, delta=0.1)
        self.assertAlmostEqual(frac_C, 0.5, delta=0.1)
        # A>C 1|1 SNP at position 241
        pos = 241
        covering_reads = fq1.coverage(pos) + fq2.coverage(pos)
        pos_bases = [r['seq'][pos - r['read_start']] for r in covering_reads]
        frac_A = pos_bases.count('A')/float(len(pos_bases))
        frac_C = pos_bases.count('C')/float(len(pos_bases))
        self.assertAlmostEqual(frac_A, 0.0, delta=0.1)
        self.assertAlmostEqual(frac_C, 1.0, delta=0.1)

    def test_integration_3(self):
        """somatic mode with mutations specified by input VCFs"""
        out1 = path_join(dirname(__file__), "test_integration_3.1.fq")
        out2 = path_join(dirname(__file__), "test_integration_3.2.fq")
        read_length = 150
        contamination = 0.3
        args = [
            "--seed", "12345678",
            "--sample-name", "c9a6be94-bdb7-4c0d-a89d-4addbf76e486",
            "--vcf-input", self.vcfgrm,
            "--somatic-mode",
            "--sample-name2", "d44d739c-0143-4350-bba5-72dd068e05fd",
            "--contamination", str(contamination),
            "--vcf-input2", self.vcfsom,
            "--num-pairs", "320",
            "--quals-from", self.qpxml,
            "--length1", str(read_length),
            "--length2", str(read_length),
            self.fa1,
            out1,
            out2]
        qasim.workflow(qasim.get_args(args))
        # see comments in test_integration_2
        fq1 = Fastq.forwardize(Fastq(out1))
        fq2 = Fastq.forwardize(Fastq(out2))
        # Verify that a germline variant is still present in the somatic reads
        # A>C 1|0 SNP at position 161
        pos = 161
        covering_reads = fq1.coverage(pos) + fq2.coverage(pos)
        pos_bases = [r['seq'][pos - r['read_start']] for r in covering_reads]
        frac_A = pos_bases.count('A')/float(len(pos_bases))
        frac_C = pos_bases.count('C')/float(len(pos_bases))
        self.assertAlmostEqual(frac_A, 0.5, delta=0.1)
        self.assertAlmostEqual(frac_C, 0.5, delta=0.1)
        # A>ACG 0|1 insertion at position 881
        pos = 881
        # We look at only "original" forward reads because in the case of the
        # first insertion specified by the somatic vcf their start coordinate
        # is unshifted from the reference, and we can perform naive position
        # arithmetic to obtain the values of the bases at pos, pos+1 & pos+2.
        #
        # Contrast this to reverse reads where the end coordinate we get
        # from the read id is /after/ the insertion and all read positions
        # relative to it are shifted by len(insert_size).
        # Considering only fwd reads makes this an imperfect test of reads
        #
        # generated over indels but a better one will require proper
        # alignment to the reference.
        fwd_covering_reads = [
            r for r in fq1.coverage(pos) + fq2.coverage(pos)
            if r['read'] == 1 and r['frag_start'] < r['frag_end'] or
            r['read'] == 2 and r['frag_start'] > r['frag_end']]
        pos1_bases = [r['seq'][pos - r['read_start']]
                      for r in fwd_covering_reads]
        pos2_bases = [r['seq'][pos + 1 - r['read_start']]
                      for r in fwd_covering_reads
                      if pos + 1 - r['read_start'] < read_length]
        pos3_bases = [r['seq'][pos + 2 - r['read_start']]
                      for r in fwd_covering_reads
                      if pos + 2 - r['read_start'] < read_length]
        frac_A1 = pos1_bases.count('A')/float(len(pos1_bases))
        self.assertAlmostEqual(frac_A1, 1.0, delta=0.1)
        frac_A2 = pos2_bases.count('A')/float(len(pos2_bases))
        self.assertAlmostEqual(frac_A2, 0.5 * (1 + contamination), delta=0.1)
        frac_C2 = pos2_bases.count('C')/float(len(pos2_bases))
        self.assertAlmostEqual(frac_C2, 0.5 * (1 - contamination), delta=0.1)
        frac_A3 = pos3_bases.count('A')/float(len(pos3_bases))
        self.assertAlmostEqual(frac_A3, 0.5 * (1 + contamination), delta=0.1)
        frac_G3 = pos3_bases.count('G')/float(len(pos3_bases))
        self.assertAlmostEqual(frac_G3, 0.5 * (1 - contamination), delta=0.1)


if __name__ == '__main__':
    unittest.main()
