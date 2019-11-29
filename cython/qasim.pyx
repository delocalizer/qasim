'''
Simulate mutations, fragment generation and shotgun sequencing on genomes.
'''
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.math cimport floor, ceil
from libc.stdio cimport FILE, fdopen, stdout, stderr, fprintf
from libc.stdlib cimport calloc, free
from cpython cimport bool
cimport numpy as np


import argparse
import numpy as np
import sys
from datetime import datetime
from xml.etree import ElementTree as ET


cdef extern from "<stdlib.h>":
    void exit(int)
    double drand48()
    void srand48(long)


cdef extern int genreads(FILE *fpout1,
                         FILE *fpout2,
                         uint8_t *s1,
                         uint8_t *s2,
                         uint32_t *rel1,
                         uint32_t *rel2,
                         uint32_t len1,
                         uint32_t len2,
                         uint64_t n_pairs,
                         int dist,
                         int std_dev,
                         int size_l,
                         int size_r,
                         double error_rate,
                         double ambig_frac,
                         const char *seqname,
                         int num_quals,
                         double ***p,
                         char ***q,
                         double **conversions,
                         int wgsim_mode)


cdef uint8_t *nst_nt4_table = [
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 5, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
]


cdef bool WGSIM_MODE = False


HELP_SOMATIC_MODE = (
    'If "-S, --somatic-mode" is specified then mutation and read generation will '
    'be run /twice/ - the first time generating "germline" mutations, the second '
    'time generating "somatic" mutations. Specifying the other options in this '
    'group has no effect if not in somatic mode.')
HELP_WGSIM_MODE = (
    'In this mode insertions are generated using the same logic as original wgsim.c'
    ' - i.e. max_insertion is set to 4, and insert bases are reversed with respect '
    'to generation order.')
HELP_FRAGMENTS = (
    'The transition/transversion rates represent the chance that the given random '
    'base conversion occurs at any position. This is applied after fragment '
    'generation but before sequencing read error, and can be used to model sample '
    'degradation, e.g. with a non-zero C>T rate for FFPE samples.')
HELP_READS = (
    'If -e is specified then a fixed error rate (and quality string) is used '
    'along the entire read. If -Q is specified then quality scores will be '
    'randomly generated according to the distributions specified in the two '
    'files, and the error rate and quality value is calculated per base. The '
    'files specified by -Q should be qprofiler-like XML documents with <QUAL> '
    'elements containing <Cycle> and <TallyItem> elements with "count" and '
    '"value" attributes.')
MSG_CTOR_SEQ_OR_SIZE = (
    'DipSeq constructor must get a sequence or size, not both.\n')
MSG_NO_SYNC_REF = (
    'A and B sequences of reference do not start at same relative position\n')
MSG_REF_NOT_HAPPY = (
    'A and B sequences of reference are not haploid at %i\'th value\n')
MSG_UNKGT = 'Unhandled genotype at %s %i %s %s %s\n'
MSG_UNKVAR = 'Unhandled variant type %s %i %s %s\n'
MSG_SKIP_MUT = (
    'Mutation at %(POS)i in allele %(allele)i is in deleted region of original, '
    'skipping...\n')
EXCEPT_MUT = (
    'Mutation specified at POS %(POS)i is unsupported, being within a previously '
    'specified deletion at POS %(OLDPOS)i.')
EXCEPT_CYCLENUM = (
    'Cycles found in file: %i, specified read length: %i: cycles must be 1 + '
    'read length')


cdef class VCF:

    cdef public str sample
    cdef public str header
    cdef public list columns
    cdef public list records

    def __cinit__(self, sample='SAMPLE'):
        self.sample = sample
        self.header = (
            "##fileformat=VCFv4.1\n"
            "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">\n"
            "##INFO=<ID=SOMATIC,Number=0,Type=Flag,Description=\"Somatic event\">\n"
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
            "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality\">\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t%s\n" %
            self.sample)
        self.columns = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL',
                        'FILTER', 'INFO', 'FORMAT', 'SAMPLE']
        self.records = []

    def tuples(self):
        '''
        Generator returning CHROM, POS, REF, ALT, GT tuples.
        Records are returned ordered by CHROM, POS, REF, ALT so at the same
        position snps are before insertions which are before deletions.
        When all records are exhausted, dummy records are returned for ever
        (CHROM='', POS=0)
        '''
        cdef dict r
        cdef int POS
        cdef str CHROM, REF, ALT, GT
        cdef object iterrec
        self.records.sort(
            key=lambda x: (
                x['#CHROM'],
                x['POS'],
                len(x['REF']),
                len(x['ALT'])))
        iterrec = iter(self.records)
        while True:
            try:
                r = iterrec.__next__()
            except StopIteration:
                r = {}
            CHROM = r.get('#CHROM', '')
            POS = r.get('POS', 0)
            REF = r.get('REF', '')
            ALT = r.get('ALT', '')
            GT = r.get('SAMPLE', '0|0')[:3]
            yield CHROM, POS, REF, ALT, GT

    def write(self, dest):
        '''
        @param dest: file or stream or anything with write and flush.
        '''
        cdef dict r
        dest.write(self.header)
        for r in self.records:
            dest.write('\t'.join(
                map(str, [r.get(col, '') for col in self.columns])) + '\n')

    @classmethod
    def fromfile(cls, filename, sample='SAMPLE'):
        '''
        Return VCF read from a vcf file
        '''
        cdef VCF v
        cdef str line
        cdef list tkns
        cdef dict r
        v = cls(sample)
        with open(filename, 'rt') as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                tkns = line.strip().split('\t')
                if len(tkns):
                    r = {v.columns[i]: tkns[i] for i in range(len(v.columns))}
                    r['POS'] = int(r['POS'])
                    v.records.append(r)
        return v


cdef class DipSeq:

    cdef public str seqid, description
    cdef public uint8_t fold              # line length for output
    cdef public uint8_t[:] seqA, seqB     # base sequence
    cdef public uint32_t[:] relA, relB    # position on reference
    cdef public uint32_t stopA, stopB

    def __cinit__(self, seqid, description, hapseq=None, size=None, fold=80):
        '''
        Parameterized constructor initialized either with a haploid (1-d) buffer
        as seq argument or empty, if supplied size.
        '''
        cdef uint32_t i
        self.seqid = seqid
        self.description = description
        self.fold = fold

        if hapseq is not None and size is not None:
            raise Exception(MSG_CTOR_SEQ_OR_SIZE)
        elif hapseq is not None:
            self.seqA = self.seqB = hapseq
            self.relA = self.relB = np.empty((self.seqA.shape[0],), 'uint32')
            self.stopA = self.stopB = self.seqA.shape[0]
            self.fold = self.stopA if self.stopA < self.fold else self.fold
            for i in range(self.stopA):
                # bases (A,C,G,T,N,-) into ints (0,1,2,3,4,5)
                self.seqA[i] = self.seqB[i] = nst_nt4_table[self.seqA[i]]
                # relative position is simply 1-based index
                self.relA[i] = self.relB[i] = i + 1
        else:
            self.seqA = np.empty((size,), 'uint8')
            self.seqB = np.empty((size,), 'uint8')
            self.relA = np.empty((size,), 'uint32')
            self.relB = np.empty((size,), 'uint32')
            self.stopA = size
            self.stopB = size

    cdef get_ptrs(self, uint8_t *seq[2], uint32_t *rel[2]):
        seq[0] = &(self.seqA[0])
        seq[1] = &(self.seqB[0])
        rel[0] = &(self.relA[0])
        rel[1] = &(self.relB[0])

    cdef get_stop(self, uint32_t stop[2]):
        stop[0] = self.stopA
        stop[1] = self.stopB

    @classmethod
    def mutagen(cls,
                DipSeq reference not None,
                VCF vcf not None,
                double mut_rate,
                double homo_frac,
                double indel_frac,
                double indel_extend,
                int max_insertion,
                bool somatic_mode=False):
        '''
        Generate mutations on haploid reference sequence, updating vcf in-place.
        '''
        cdef int allele, deleting = 0, appended = 0
        cdef uint8_t refbase, prevref, snp, j
        cdef uint8_t *refseq[2]
        cdef uint32_t i, POS
        cdef uint32_t rstop[2]
        cdef uint32_t *refrel[2]
        cdef str seqid = reference.seqid
        cdef list records = vcf.records, alt
        cdef dict vrec = {}
        cdef dict homrec = {'ID': '.', 'QUAL': '.', 'FILTER': 'PASS', 'INFO': '',
                            'FORMAT': 'GT:GQ', 'SAMPLE': '1|1:200'}
        cdef dict hetrec = {'ID': '.', 'QUAL': '.', 'FILTER': 'PASS', 'INFO': '',
                            'FORMAT': 'GT:GQ'}

        if somatic_mode:
            homrec['INFO'] += 'SOMATIC;'
            hetrec['INFO'] += 'SOMATIC;'

        reference.get_ptrs(refseq, refrel)
        reference.get_stop(rstop)

        if refrel[0][0] != refrel[1][0]:
            raise Exception(MSG_NO_SYNC_REF)

        # start mutating at second base for convenience so we don't have to handle
        # vcf requirement that REF be _after_ the variant when variant is at POS=1
        prevref = refseq[0][0]

        for i in range(1, rstop[0]):

            if (refseq[0][i] != refseq[1][i] or
                refrel[0][i] != refrel[1][i] or
                refrel[0][i] != i + 1):
                raise Exception(MSG_REF_NOT_HAPPY % i)

            POS = i + 1
            refbase = refseq[0][i]

            if deleting:
                if drand48() < indel_extend:
                    vrec['REF'] += "ACGTN"[refbase]
                    prevref = refbase
                    continue
                else:
                    records.append(vrec)
                    appended = 1
                    deleting = 0

            if refbase < 4 and drand48() < mut_rate:
                vrec = {
                    '#CHROM': seqid,
                    'POS': POS,
                    'REF': "ACGT"[refbase],
                    'ALT': ''}
                appended = 0

                if drand48() >= indel_frac:                         # =SNP=
                    snp = (refbase + <uint8_t>(drand48() * 3.0 + 1)) & 3
                    vrec['ALT'] = "ACGT"[snp]
                    if drand48() < homo_frac:                       # hom SNP
                        vrec.update(homrec)
                    else:                                           # het SNP
                        allele = 0 if drand48() < 0.5 else 1
                        hetrec['SAMPLE'] = '%s|%s:200' % (1 - allele, allele)
                        vrec.update(hetrec)
                    records.append(vrec)
                    appended = 1

                else:
                    if drand48() < 0.5:                             # =DEL=
                        deleting = 1
                        vrec.update({'POS': POS - 1,
                                     'REF': "ACGTN"[prevref] + "ACGT"[refbase],
                                     'ALT': "ACGTN"[prevref]})
                        if drand48() < homo_frac:                   # hom DEL
                            vrec.update(homrec)
                        else:                                       # het DEL
                            allele = 0 if drand48() < 0.5 else 1
                            hetrec['SAMPLE'] = '%s|%s:200' % (
                                1 - allele, allele)
                            vrec.update(hetrec)

                    else:                                           # =INS=
                        alt = insertion(refbase, indel_extend, max_insertion)
                        vrec['ALT'] = ''.join(["ACGT"[j] for j in alt])
                        if drand48() < homo_frac:                   # hom INS
                            vrec.update(homrec)
                        else:                                       # het INS
                            allele = 0 if drand48() < 0.5 else 1
                            hetrec['SAMPLE'] = '%s|%s:200' % (
                                1 - allele, allele)
                            vrec.update(hetrec)
                        records.append(vrec)
                        appended = 1

            prevref = refbase

        # any unfinished deletion
        if vrec.get('#CHROM') and not appended:
            records.append(vrec)

    def transform(self,
                  DipSeq original not None,
                  VCF vcf):
        '''
        Apply mutations from a vcf to a diploid sequence, updating self in place.
        For insertions and deletions, the REF and ALT sequences respectively
        must be a single base only.
        '''
        # gt is coded phased genotype 0=(0|0), 1=(1|0), 2=(0|1), 3=(1|1)
        cdef int allele, c, snp, ins, gt = 0, refsz, altsz
        cdef uint8_t *origseq[2]
        cdef uint8_t *mutseq[2]
        cdef uint32_t POS, OLDPOS, del_l, del_r, r
        cdef uint32_t opos[2]
        cdef uint32_t mpos[2]
        cdef uint32_t ostop[2]
        cdef uint32_t mstop[2]
        cdef uint32_t *origrel[2]
        cdef uint32_t *mutrel[2]
        cdef str alt, seqid = original.seqid.split('.')[0]
        cdef str CHROM, REF, ALT, GT
        cdef object mutations

        original.get_ptrs(origseq, origrel)
        original.get_stop(ostop)
        self.get_ptrs(mutseq, mutrel)
        self.get_stop(mstop)

        # apply mutations to two original alleles independently
        opos[0] = opos[1] = 0
        mpos[0] = mpos[1] = 0
        for allele in range(2):

            # at given location tuples() returns snps, insertions then
            # deletions
            mutations = vcf.tuples()
            CHROM, POS, REF, ALT, GT = mutations.__next__()
            del_l = del_r = 0
            OLDPOS = 0
            while opos[allele] < ostop[allele]:

                c = origseq[allele][opos[allele]]
                r = origrel[allele][opos[allele]]

                # in deletion
                if (del_l <= <uint32_t>r <= del_r):
                    if r == POS:
                        # DipSeq.mutagen() doesn't generate mutations within
                        # deletions with the exception that the reference base
                        # of an indel may be the last base of previous deletion
                        if not (len(REF) != len(ALT) and POS == del_r):
                            raise Exception(EXCEPT_MUT %
                                            {'POS': POS, 'OLDPOS': OLDPOS})
                    else:
                        opos[allele] += 1
                        continue

                # process any mutation at this location
                if CHROM == seqid and POS == r:
                    if GT == '1|0':                      # het (1|0)
                        gt = 1
                    elif GT == '0|1':                    # het (0|1)
                        gt = 2
                    elif GT == '1|1':                    # hom
                        gt = 3
                    else:
                        raise Exception(
                            MSG_UNKGT %
                            (seqid, opos[allele], REF, ALT, GT))

                    refsz, altsz = len(REF), len(ALT)
                    if refsz == altsz == 1:              # snp
                        snp = nst_nt4_table[ord(ALT[0])]
                        if gt == 3 or gt == allele + 1:  # hit allele
                            update(allele, mutseq, snp, mutrel, r, mpos, mstop)
                        else:
                            update(allele, mutseq, c, mutrel, r, mpos, mstop)

                    elif altsz > refsz and refsz == 1:   # insertion
                        if ((del_l <= r <= del_r) or POS == OLDPOS):
                            # continued deletion or collocated snp
                            ALT = ALT[1:]
                        if gt == 3 or gt == allele + 1:  # hit allele
                            for alt in ALT:
                                ins = nst_nt4_table[ord(alt)]
                                update(
                                    allele, mutseq, ins, mutrel, r, mpos, mstop)
                        elif not ((del_l <= r <= del_r) or POS == OLDPOS):
                            update(allele, mutseq, c, mutrel, r, mpos, mstop)

                    elif altsz < refsz and altsz == 1:   # deletion
                        if not ((del_l <= r <= del_r) or POS == OLDPOS):
                            # no continued deletion or collocated snp/insertion
                            update(allele, mutseq, c, mutrel, r, mpos, mstop)
                        if gt == 3 or gt == allele + 1:  # hit allele
                            del_l = POS + 1
                            del_r = POS + len(REF) - 1

                    OLDPOS = POS
                    CHROM, POS, REF, ALT, GT = mutations.__next__()
                    # keep processing at same location
                    if POS == OLDPOS:
                        continue
                    else:
                        opos[allele] += 1

                else:
                    update(allele, mutseq, c, mutrel, r, mpos, mstop)
                    opos[allele] += 1

                # get another mutation if new location is past current POS
                while CHROM == seqid and origrel[allele][
                        opos[allele]] > POS and POS > 0:
                    sys.stderr.write(MSG_SKIP_MUT %
                                     {'allele': allele, 'POS': POS})
                    CHROM, POS, REF, ALT, GT = mutations.__next__()

        self.stopA = mpos[0]
        self.stopB = mpos[1]

    def write(self, dest):
        '''
        Output sequences from a DipSeq instance.
        Only the last digit of the reference-relative coordinate is shown.

        @param dest: file or stream or anything with write and flush.
        '''
        cdef int startln, allele, j, fold = self.fold
        cdef uint8_t *seq[2]
        cdef uint32_t i
        cdef uint32_t stop[2]
        cdef uint32_t *rel[2]

        self.get_ptrs(seq, rel)
        self.get_stop(stop)

        for allele in range(2):
            startln = 0
            dest.write('>%s.%s %s\n' % (self.seqid, allele, self.description))
            for i in range(stop[allele]):
                dest.write("ACGTN"[seq[allele][i]])
                if i and not (rel[allele][i]) % fold:
                    dest.write('\n')
                    for j in range(startln, i + 1):
                        dest.write(str(rel[allele][j] % 10))
                    dest.write('\n')
                    startln = i + 1
        dest.flush()


cdef inline int update(int allele,
                       uint8_t *seq[2],
                       uint8_t seqval,
                       uint32_t *rel[2],
                       uint32_t relval,
                       uint32_t *pos,
                       uint32_t *stop) nogil:
    '''
    Helper to update DipSeq sequences and position marker.
    '''
    if pos[allele] >= stop[allele]:
        fprintf(
            stderr,
            "Array bounds exceeded. Insertions too large/too many? "
            "Try smaller values of -X, -r, and/or -R")
        return 1
    seq[allele][pos[allele]] = seqval
    rel[allele][pos[allele]] = relval
    pos[allele] += 1
    return 0


cdef insertion(uint8_t first,
               double indel_extend,
               int max_insertion):
    '''
    Build and return list of randomly inserted characters as bytes in vcf 'ALT'
    format i.e. with first character as reference base. If WGSIM_MODE is set,
    the generated insertion will be reversed with respect to generation order.
    '''
    cdef int num_ins = 0
    cdef list ins
    ins = [] if WGSIM_MODE else [first, ]
    while True:
        num_ins += 1
        ins.append(<uint8_t>(drand48() * 4.0))
        if num_ins >= max_insertion or drand48() > indel_extend:
            if WGSIM_MODE:
                # reverse for consistency with bitshift method
                ins.append(first)
                return ins[::-1]
            else:
                return ins


def read_fasta(filename):
    '''
    Read from file, yield FASTA sequences.
    '''
    cdef str seqid = '', description = ''
    cdef object seqb = bytearray()
    cdef int fold = 80
    cdef bool firstseq = True
    cdef bool firstseqline = True

    with open(filename, 'rb') as fh:

        for line in fh:
            l = line[:-1]           # strip newline
            if l == b'':            # blank line
                continue
            elif l[0] in (59, 62):  # ';', '>'
                f = DipSeq(seqid, description, seqb, fold=fold)
                tkns = l.decode('ASCII').lstrip('>').split(' ', 1)
                seqid = tkns[0] if len(tkns) >= 1 else ''
                description = tkns[1] if len(tkns) == 2 else ''
                seqb = bytearray()
                firstseqline = True
                if firstseq:
                    firstseq = False
                else:
                    yield f
            else:
                if firstseqline:
                    firstseqline = False
                    fold = len(l)
                seqb.extend(l)
        # EOF
        yield(DipSeq(seqid, description, seqb, fold=fold))


def gen_quals(str filename, int readlen, int num_quals, char[:, :] qvals, double[:, :] pvals):
    '''
    Generate quality values from a file
    '''
    cdef int i, j
    cdef object doc, root, Q, cumdist
    cdef list cycles, tallies
    cdef np.ndarray qarray

    doc = ET.parse(filename)
    root = doc.getroot()
    Q = root.find('.//QUAL')
    cycles = Q.findall('.//Cycle')

    if len(cycles) != readlen + 1:
        raise Exception(EXCEPT_CYCLENUM % (len(cycles), readlen))

    tallies = []
    for c in cycles:
        cyclenum = int(c.get('value'))
        ctallies = sorted(
            (cyclenum, int(t.get('value')), int(t.get('count')))
            for t in c.findall('TallyItem'))
        tallies.extend(ctallies)

    cumdist = {}
    for cycle, qual, count in tallies:
        cumdist.setdefault(cycle, [])
        prev = cumdist[cycle][-1][1] if len(cumdist[cycle]) else 0
        new = prev + count
        cumdist[cycle].append((qual, new))

    for i in range(num_quals):
        for j in range(readlen):
            qvals[i, j] = _randqual(cumdist, j + 1)
            pvals[i, j] = 10 ** (qvals[i, j] / -10.0)  # Phred definition


cdef inline int _randqual(dict dist, int cycle):
    '''
    Return a random, representative quality value for a cycle given a dict of
    lists of tuples representing cumulative distribution function of qualities
    per cycle:{ <cycle1>: [(<qual1>, <cumulative count1>),
                           (<qual2>, <cumulative count2>),
                           ...],
    '''
    cdef long maxcount, count, rand
    cdef int qual
    maxcount = dist[cycle][-1][1]
    rand = <long>(drand48() * maxcount)
    for qual, count in dist[cycle]:
        if count > rand:
            return qual
    return 0


def _t_randqual(dist, cycle):
    '''
    Shim for unit testing the cdef inline method _randqual
    '''
    return _randqual(dist, cycle)


def reseed(seed):
    '''
    Hook for (re)seeding the random number generator
    '''
    srand48(seed)


def get_args(argv):
    '''
    Return options and arguments
    '''
    class Range(object):

        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __eq__(self, other):
            return self.start <= other <= self.end

        def __repr__(self):
            return "Range %s<=<%s" % (str(self.start), str(self.end))

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('fasta', help='Reference FASTA')
    p.add_argument('read1fq', help='Output file for read1', type=argparse.FileType('wb'))
    p.add_argument('read2fq', help='Output file for read2', type=argparse.FileType('wb'))

    mutgrp = p.add_argument_group('Mutations')
    mutgrp.add_argument('-r', '--mut-rate', help='mutation rate', type=float, default=0.001, choices=[Range(0.0,1.0)])
    mutgrp.add_argument('-H', '--homo-frac', help='fraction of mutations that are homozygous', type=float, default=0.333333, choices=[Range(0.0,1.0)])
    mutgrp.add_argument('-R', '--indel-frac', help='fraction of mutations that are indels', type=float, default=0.15, choices=[Range(0.0,1.0)])
    mutgrp.add_argument('-X', '--indel-extend', help='probability an indel is extended', type=float, default=0.3, choices=[Range(0.0,1.0)])
    mutgrp.add_argument('-M', '--max-insertion', help='Maximum size of generated insertions (regardless of -X value)', type=int, default=1000)
    mutgrp.add_argument('-n', '--sample-name', help='name of sample for vcf output', type=str, default='SAMPLE')

    mutgrpio = mutgrp.add_mutually_exclusive_group(required=True)
    mutgrpio.add_argument('-o', '--output', metavar='VCF', help='output generated mutations to file', type=argparse.FileType('wt'))
    mutgrpio.add_argument('-V', '--vcf-input', help='use input vcf file as source of mutations instead of randomly generating them', type=str)

    mutgrp2 = p.add_argument_group(title='Somatic mutations', description=HELP_SOMATIC_MODE)
    mutgrp2.add_argument('-S', '--somatic-mode', action='store_true')
    mutgrp2.add_argument('--mut-rate2', help='somatic mutation rate', type=float, default=0.000001, choices=[Range(0.0,1.0)])
    mutgrp2.add_argument('--homo-frac2', help='fraction of somatic mutations that are homozygous', type=float, default=0.333333, choices=[Range(0.0,1.0)])
    mutgrp2.add_argument('--indel-frac2', help='fraction of somatic mutations that are indels', type=float, default=0.15, choices=[Range(0.0,1.0)])
    mutgrp2.add_argument('--indel-extend2', help='probability a somatic indel is extended', type=float, default=0.3, choices=[Range(0.0,1.0)])
    mutgrp2.add_argument('--max-insertion2', help='Maximum size of generated somatic insertions (regardless of -X value)', type=int, default=1000)
    mutgrp2.add_argument('--contamination', help='fraction of reads generated from "germline" sequence', type=float, default=0.0, choices=[Range(0.0,1.0)])
    mutgrp2.add_argument('--sample-name2', help='name of sample for vcf2 output', type=str, default='SOMATIC')

    mutgrp2io = mutgrp2.add_mutually_exclusive_group()
    mutgrp2io.add_argument('--output2', metavar='VCF2', help='output generated somatic mutations to file', type=argparse.FileType('wt'))
    mutgrp2io.add_argument('--vcf-input2', help='use input vcf file as source of somatic mutations instead of randomly generating them', type=str)

    fragrp = p.add_argument_group('Fragments', description=HELP_FRAGMENTS)
    fragrp.add_argument('-z', '--size', help='mean fragment size', type=int, default=500)
    fragrp.add_argument('-s', '--std-dev', help='fragment standard deviation', type=int, default=50)
    fragrp.add_argument('--AC', help='A>C transversion rate', type=float, choices=[Range(0.0,1.0)])
    fragrp.add_argument('--AG', help='A>G transition rate', type=float, choices=[Range(0.0,1.0)])
    fragrp.add_argument('--AT', help='A>T transversion rate', type=float, choices=[Range(0.0,1.0)])
    fragrp.add_argument('--CA', help='C>A transversion rate', type=float, choices=[Range(0.0,1.0)])
    fragrp.add_argument('--CG', help='C>G transversion rate', type=float, choices=[Range(0.0,1.0)])
    fragrp.add_argument('--CT', help='C>T transition rate', type=float, choices=[Range(0.0,1.0)])
    fragrp.add_argument('--GA', help='G>A transition rate', type=float, choices=[Range(0.0,1.0)])
    fragrp.add_argument('--GC', help='G>C transversion rate', type=float, choices=[Range(0.0,1.0)])
    fragrp.add_argument('--GT', help='G>T transversion rate', type=float, choices=[Range(0.0,1.0)])
    fragrp.add_argument('--TA', help='T>A transversion rate', type=float, choices=[Range(0.0,1.0)])
    fragrp.add_argument('--TC', help='T>C transition rate', type=float, choices=[Range(0.0,1.0)])
    fragrp.add_argument('--TG', help='T>G transversion rate', type=float, choices=[Range(0.0,1.0)])

    rdsgrp = p.add_argument_group('Reads', description=HELP_READS)
    rdsgrp.add_argument('-N', '--num-pairs', help='number of read pairs', type=int, default=1000000)
    rdsgrp.add_argument('-1', '--length1', help='length of read 1', type=int, default=100)
    rdsgrp.add_argument('-2', '--length2', help='length of read 2', type=int, default=100)
    rdsgrp.add_argument('-A', '--ambig-frac', help='discard read if fraction of "N" bases exceeds this', type=float, default=1.0, choices=[Range(0.0,1.0)])

    rdsgrperr = rdsgrp.add_mutually_exclusive_group()
    rdsgrperr.add_argument('-e', '--error-rate', help='read error rate (constant)', type=float, default=0.002, choices=[Range(0.0,1.0)])
    rdsgrperr.add_argument('-Q', '--quals-from', help='generate random quality strings for read 1 and read 2 respectively from the distributions specified in the files', type=str, nargs=2, metavar=('R1_QUALS', 'R2_QUALS'))
    # --num-quals is part of rdsgrp but leave it here after --quals-from
    # so it's grouped logically in -h, --help output
    rdsgrp.add_argument('--num-quals', help='number of quality strings to generate from distribution files', type=int, default=10000)

    othgrp = p.add_argument_group('Other')
    othgrp.add_argument('-d', '--seed', help='seed for random generator (default=current time)', type=int, default=datetime.now().strftime('%s'))
    othgrp.add_argument('-t', '--test-output', help='print mutated sequences to stdout', action='store_true')
    othgrp.add_argument('-w', '--wgsim-mode', help=HELP_WGSIM_MODE, action='store_true')

    args = p.parse_args(argv)

    global WGSIM_MODE
    WGSIM_MODE = args.wgsim_mode
    if WGSIM_MODE:
        args.max_insertion = 4
        args.max_insertion2 = 4
    del args.wgsim_mode

    if args.somatic_mode and not args.vcf_input2 and not args.output2:
        p.error(
            'one of --output2 or --vcf-input2 is required when specifying somatic mode')

    return args


def workflow(args):
    '''
    Run the workflow specified by the command line args.
    '''

    # variables from command line args
    cdef str fasta = args.fasta, sample_name = args.sample_name, \
        sample_name2 = args.sample_name2, vcf_input = args.vcf_input, \
        vcf_input2 = args.vcf_input2, r1_quals, r2_quals
    cdef object read1fq = args.read1fq, read2fq = args.read2fq, \
        output = args.output, output2 = args.output2,
    cdef int max_insertion = args.max_insertion, size = args.size, \
        std_dev = args.std_dev, num_pairs = args.num_pairs, \
        length1 = args.length1, length2 = args.length2, \
        num_quals = args.num_quals, seed = args.seed, \
        max_insertion2 = args.max_insertion2
    cdef double mut_rate = args.mut_rate, homo_frac = args.homo_frac, \
        indel_frac = args.indel_frac, indel_extend = args.indel_extend, \
        mut_rate2 = args.mut_rate2, homo_frac2 = args.homo_frac2, \
        indel_frac2 = args.indel_frac2, indel_extend2 = args.indel_extend2, \
        contamination = args.contamination, error_rate = args.error_rate, \
        ambig_frac = args.ambig_frac
    cdef bool somatic_mode = args.somatic_mode, test_output = args.test_output

    cdef uint64_t tot_len = 0, n_pairs, n_som, n_grm
    cdef int n_ref = 0, i, mutseqsize, mutseq2size
    cdef VCF vcf, vcf2
    cdef DipSeq refseq, mutseq, mutseq2
    cdef FILE *fpout1
    cdef FILE *fpout2
    cdef uint8_t[:, :, ::1] qvals
    cdef double[:, :, ::1] pvals
    cdef double *conversions[4]
    cdef char **q[2]
    cdef double **p[2]

    sys.stderr.write("[%s] seed = %i\n" % (__name__, seed))
    reseed(seed)

    fpout1 = fdopen(read1fq.fileno(), 'wb')
    fpout2 = fdopen(read2fq.fileno(), 'wb')

    if vcf_input:
        sys.stderr.write(
            "[%s] reading mutations from file %s (1)\n" %
            (__name__, vcf_input))
        vcf = VCF.fromfile(vcf_input, sample_name)
    else:
        vcf = VCF(sample_name)

    if somatic_mode:
        if vcf_input2:
            sys.stderr.write(
                "[%s] reading mutations from file %s (2)\n" %
                (__name__, vcf_input2))
            vcf2 = VCF.fromfile(vcf_input2, sample_name2)
        else:
            vcf2 = VCF(sample_name2)

    if args.quals_from:
        r1_quals, r2_quals = args.quals_from
        qvals = np.ndarray((2, num_quals, max(length1, length2)), order='C', dtype='u1')
        pvals = np.ndarray((2, num_quals, max(length1, length2)), order='C')
        for i, (quals_file, read_length) in \
            enumerate(zip([r1_quals, r2_quals], [length1, length2])):
            sys.stderr.write(
                "[%s] generating qualities for read %s from %s\n" %
                (__name__, i + 1, quals_file))
            reseed(seed)
            gen_quals(quals_file, read_length, num_quals, qvals[i], pvals[i])
            q[i] = <char**>calloc(num_quals, sizeof(char*))
            p[i] = <double**>calloc(num_quals, sizeof(double*))
            for j in range(num_quals):
                q[i][j] = <char*>&qvals[i, j, 0]
                p[i][j] = <double*>&pvals[i, j, 0]
    else:
        # special value to indicate to genreads() to use fixed error rate
        num_quals = 0

    for i in range(4):                                                          
        conversions[i] = <double*>calloc(4, sizeof(double))   
    # use -1 as unambiguous 'not set' value
    conversions[0][1] = -1 if args.AC is None else args.AC
    conversions[0][2] = -1 if args.AG is None else args.AG
    conversions[0][3] = -1 if args.AT is None else args.AT
    conversions[1][0] = -1 if args.CA is None else args.CA
    conversions[1][2] = -1 if args.CG is None else args.CG
    conversions[1][3] = -1 if args.CT is None else args.CT
    conversions[2][0] = -1 if args.GA is None else args.GA
    conversions[2][1] = -1 if args.GC is None else args.GC
    conversions[2][3] = -1 if args.GT is None else args.GT
    conversions[3][0] = -1 if args.TA is None else args.TA
    conversions[3][1] = -1 if args.TC is None else args.TC
    conversions[3][2] = -1 if args.TG is None else args.TG

    for refseq in read_fasta(fasta):
        n_ref += 1
        tot_len += refseq.stopA
    sys.stderr.write("[%s] %d input sequences, total length: %i\n" %
                     (__name__, n_ref, tot_len))

    # iterate through input sequences
    for refseq in read_fasta(fasta):
        n_pairs = <uint64_t>(<float>refseq.stopA / <float>tot_len * num_pairs)
        n_grm = <uint64_t>floor(n_pairs * contamination) if somatic_mode else n_pairs
        n_som = <uint64_t>ceil(n_pairs * (1 - contamination))

        if refseq.stopA < <uint32_t>(size + 3 * std_dev):
            sys.stderr.write(
                "[%s] skip sequence '%s' as it is shorter than %d\n" %
                (__name__, refseq.seqid, size + 3 * std_dev))
            continue

        # this size is a bit arbitrary, but seems to cope with most cases
        mutseqsize = <int>(1.1 * refseq.seqA.shape[0] + 10 * max_insertion)
        mutseq = DipSeq(refseq.seqid + '.1',
                        refseq.description + '.1',
                        size=mutseqsize,
                        fold=refseq.fold)

        if not vcf_input:
            sys.stderr.write("[%s] generating mutations (1)\n" % (__name__,))
            DipSeq.mutagen(refseq,
                           vcf,
                           mut_rate,
                           homo_frac,
                           indel_frac,
                           indel_extend,
                           max_insertion)
        mutseq.transform(refseq, vcf)
        if test_output:
            mutseq.write(sys.stdout)

        # sample "germline" sequence
        bseqid = bytes(mutseq.seqid, 'ASCII')
        sys.stderr.write("[%s] generating %i reads from sequence %s\n" %
                         (__name__, n_grm, mutseq.seqid))
        if n_grm:
            # reseed(seed) # reseed during testing
            genreads(fpout1, fpout2, &(mutseq.seqA[0]), &(mutseq.seqB[0]),
                     &(mutseq.relA[0]), &(mutseq.relB[0]),
                     mutseq.stopA, mutseq.stopB,
                     n_grm, size, std_dev, length1, length2,
                     error_rate, ambig_frac, bseqid, num_quals, p, q,
                     conversions, int(WGSIM_MODE))

        if somatic_mode:
            mutseq2size = <int>(1.1 * mutseq.seqA.shape[0] + 10 * max_insertion2)
            mutseq2 = DipSeq(refseq.seqid + '.2',
                             refseq.description + '.2',
                             size=mutseq2size,
                             fold=refseq.fold)

            if not vcf_input2:
                sys.stderr.write(
                    "[%s] generating mutations (2)\n" %
                    (__name__,))
                DipSeq.mutagen(refseq,
                               vcf2,
                               mut_rate2,
                               homo_frac2,
                               indel_frac2,
                               indel_extend2,
                               max_insertion2,
                               True)
            mutseq2.transform(mutseq, vcf2)
            if test_output:
                mutseq2.write(sys.stdout)

            # sample somatic sequence
            bseqid = bytes(mutseq2.seqid, 'ASCII')
            sys.stderr.write("[%s] generating %i reads from sequence %s\n" %
                             (__name__, n_som, mutseq2.seqid))
            if n_som:
                # reseed(seed) # reseed during testing
                genreads(fpout1, fpout2, &(mutseq2.seqA[0]), &(mutseq2.seqB[0]),
                         &(mutseq2.relA[0]), &(mutseq2.relB[0]),
                         mutseq2.stopA, mutseq2.stopB,
                         n_som, size, std_dev, length1, length2,
                         error_rate, ambig_frac, bseqid, num_quals, p, q,
                         conversions, int(WGSIM_MODE))

    if not vcf_input:
        vcf.write(output)
    if somatic_mode and not vcf_input2:
        vcf2.write(output2)

    if output:
        output.close()
    if output2:
        output2.close()

    read1fq.close()
    read2fq.close()

    if args.quals_from:
        for i in range(2):
            free(q[i])
            free(p[i])

    return 0
