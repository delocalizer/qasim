"""Utilities for tests"""

import sys

BAD_ID = "line %s: id %s doesn't start with '@'"
BAD_SEQLEN = "line %s: %s is not the same length as the first read (%s)"
BAD_BASES = "line %s:  %s is not in allowed set of bases %s"
BAD_PLUS = "line %s: expected '+', got %s"
BAD_QUALS = "line %s: %s is not the same length as the first read (%s)"
MSG_INCOMPLETE = "incomplete record at end of file %s"

class Fastq:
    """A convenient data structure"""
    # This saves us importing a whole dependency on e.g. pysam just for tests

    allowed_bases = {'A', 'C', 'G', 'T', 'N'}

    def __init__(self, filename):

        self.records = []
        self.readlength = -1

        with open(filename, 'rt') as fh:
            lastline = 0
            for linenum, line in enumerate(fh.readlines(), 1):
                lastline = linenum
                if linenum % 4 == 1:
                    read_id = line.strip()
                    assert read_id.startswith("@"), BAD_ID % (linenum, read_id)
                elif linenum % 4 == 2:
                    seq = line.strip()
                    if self.readlength == -1:
                        self.readlength = len(seq)
                    else:
                        assert len(seq) == self.readlength, \
                            BAD_SEQLEN % (linenum, seq, self.readlength)
                    disallowed = set(seq) - self.allowed_bases
                    assert not disallowed, \
                        BAD_BASES % (linenum, disallowed, self.allowed_bases)
                elif linenum % 4 == 3:
                    plus = line.strip()
                    assert plus == "+", BAD_PLUS % (linenum, plus)
                if linenum % 4 == 0:
                    quals = line.strip()
                    assert len(quals) == self.readlength, \
                        BAD_QUALS % (linenum, quals, self.readlength)
                    self.records.append(
                        {"id": read_id, "seq": seq, "quals": quals})
            assert lastline % 4 == 0, MSG_INCOMPLETE % (filename)
