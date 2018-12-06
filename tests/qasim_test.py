import sys
import tempfile
import unittest

from contextlib import contextmanager
from io import StringIO
from os.path import join as path_join
from os.path import dirname

from qasim.qasim import *


# test resources are located in the current dir
testfasta = path_join(dirname(__file__), 'test.fa')
test0vcf = path_join(dirname(__file__), 'test0.vcf')
test1vcf = path_join(dirname(__file__), 'test1.vcf')
test2vcf = path_join(dirname(__file__), 'test2.vcf')
test3grmvcf = path_join(dirname(__file__), 'test3.1.vcf')
test3somvcf = path_join(dirname(__file__), 'test3.2.vcf')
test4vcf = path_join(dirname(__file__), 'test4.vcf')


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
            self.assertEqual(''.join(lines[6:]),
                '\n'.join(
                    '\t'.join(str(r[c]) for c in self.test0vcf.columns)
                        for r in self.test0vcf.records
                    ) + '\n'
                ) 

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
        self.assertEqual(list(d.seqA), [0,1,2,3,4,5])

    def test_ctor_seq_and_size(self):
        with self.assertRaisesRegex(Exception, MSG_CTOR_SEQ_OR_SIZE):
            d = DipSeq("T", "TEST", bytearray([65]), 1)

    def test_print_seq(self):
        d = DipSeq("T", "TEST", hapseq=bytearray([65, 67, 71, 84, 78]))
        with captured_output() as (out, err):
            d.print_seq()
            self.assertEqual(out.getvalue(), (
                ">T.0 TEST\n"
                "ACGTN\n"
                "12345\n"
                ">T.1 TEST\n"
                "ACGTN\n"
                "12345\n"))

    def test_transform_0(self):
        """germline het & hom snps"""
        vcf = VCF.fromfile(test0vcf, "sample1")
        refseq = next(read_fasta(testfasta))
        mutseq = DipSeq(refseq.seqid + '.mut',
                refseq.description,
                size = refseq.seqA.shape[0] * 2,
                fold = refseq.fold)
        mutseq.transform(refseq, vcf)
        for i in range(refseq.stopA):
            POS = i + 1 # VCF coords
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
        refseq = next(read_fasta(testfasta))
        mutseq = DipSeq(refseq.seqid + '.mut',
                refseq.description,
                size = refseq.seqA.shape[0] * 2,
                fold = refseq.fold)
        mutseq.transform(refseq, vcf)
        insertion_1 = range(4, 7)
        self.assertEqual(''.join(base(mutseq.seqA[POS-1]) for POS in insertion_1), "AGG")
        self.assertEqual(list(mutseq.relA[POS-1] for POS in insertion_1), [4,4,4])
        insertion_2 = range(5, 9)
        self.assertEqual(''.join(base(mutseq.seqB[POS-1]) for POS in insertion_2), "CTTT")
        self.assertEqual(list(mutseq.relB[POS-1] for POS in insertion_2), [5,5,5,5])
        deletion_1A = range(15, 17)
        deletion_1B = range(16, 18)
        self.assertEqual(''.join(base(mutseq.seqA[POS-1]) for POS in deletion_1A), "CC")
        self.assertEqual(list(mutseq.relA[POS-1] for POS in deletion_1A), [13, 16])
        self.assertEqual(''.join(base(mutseq.seqB[POS-1]) for POS in deletion_1B), "CC")
        self.assertEqual(list(mutseq.relB[POS-1] for POS in deletion_1B), [13, 16])

    def test_transform_2(self):
        """complex overlapping mutations"""
        vcf = VCF.fromfile(test2vcf, "sample1")
        refseq = next(read_fasta(testfasta))
        mutseq = DipSeq(refseq.seqid + '.mut',
                refseq.description,
                size = refseq.seqA.shape[0] * 2,
                fold = refseq.fold)
        mutseq.transform(refseq, vcf)
        with captured_output() as (out, err):
            mutseq.print_seq()
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
        refseq = next(read_fasta(testfasta))

        grmseq = DipSeq(refseq.seqid + '.grm',
                refseq.description,
                size = refseq.seqA.shape[0] * 2,
                fold = refseq.fold)
        grmseq.transform(refseq, grmvcf)
        with captured_output() as (out, err):
            grmseq.print_seq()
            self.assertEqual(out.getvalue(), (
                ">TEST.grm.0 small fasta for testing\n"
                "AAAAGGCCCCAAAACCCC\n"
                "123444567890123456\n"
                ">TEST.grm.1 small fasta for testing\n"
                "AAAAGGCCAAAACCCC\n"
                "1234447890123456\n"))

        somseq = DipSeq(refseq.seqid + '.som',
                refseq.description,
                size = refseq.seqA.shape[0] * 2,
                fold = refseq.fold)
        with captured_output() as (out, err):
            expected_msg = MSG_SKIP_MUT % {'allele': 1, 'POS': 5}
            somseq.transform(grmseq, somvcf)
            self.assertEqual(err.getvalue(), expected_msg)
            somseq.print_seq()
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
        refseq = next(read_fasta(testfasta))
        mutseq = DipSeq(refseq.seqid + '.mut',
                refseq.description,
                size = refseq.seqA.shape[0] * 2,
                fold = refseq.fold)
        expected_msg = EXCEPT_MUT % {'POS': 7, 'OLDPOS': 5}
        with self.assertRaisesRegex(Exception, expected_msg):            
            mutseq.transform(refseq, vcf)


class TestQasim(unittest.TestCase):

    def test_read_fasta(self):
        with captured_output() as (out, err):
            for seq in read_fasta(testfasta):
                seq.print_seq()
                self.assertEqual(out.getvalue(), (
                    ">TEST.0 small fasta for testing\n"
                    "AAAACCCCAAAACCCC\n"
                    "1234567890123456\n"
                    ">TEST.1 small fasta for testing\n"
                    "AAAACCCCAAAACCCC\n"
                    "1234567890123456\n"))

if __name__ == '__main__':
    unittest.main()
