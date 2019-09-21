"""Utilities for tests"""

import copy
import re

BAD_ID = "line %s: id '%s' doesn't match '%s'"
BAD_SEQLEN = "line %s: %s is not the same length as the first read (%s)"
BAD_BASES = "line %s:  %s is not in allowed set of bases %s"
BAD_PLUS = "line %s: expected '+', got %s"
BAD_QUALS = "line %s: %s is not the same length as the first read (%s)"
MSG_INCOMPLETE = "incomplete record at end of file %s"


class Fastq:
    """A convenient data structure for handling the fastqs generated by qasim.
    NOTES:
    * Read id's are the form: @NAME_COORD1_COORD2_ERR1_ERR2_N/[1|2].
    * COORD1 and COORD2 are the coordinates of the fragment ends.
    * Illumina pair-end reads have read 1 forward and read 2 reverse:
                                   >>>>>>>>>>>>>>
                                                      <<<<<<<<<<<<<<
    * When run in normal (non-wgsim) mode, for pairs where read 1 is from
      the reference strand the coordinates are ordered such that:
      COORD1 < COORD2. For for "flipped" reads where read 1 is from the
      reverse strand the coordinates are ordered such that:
      COORD1 > COORD2.
    * When run in legacy (wgsim) mode, coordinates are always ordered:
      COORD1 < COORD2 and there's no way to tell by inspection what strand
      a read is from."""

    allowed_bases = {'A', 'C', 'G', 'T', 'N'}
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    id_regex = re.compile(
        r"^@(.+)_(\d+)_(\d+)_e(\d+)_e(\d+)_([a-f0-9]+)\/([12])$")

    def __init__(self, filename):

        self.records = []
        self.read_length = -1
        self.forwardized = False
        self.minpos = -1
        self.maxpos = -1

        with open(filename, 'rt') as fh:
            read = frag_start = frag_end = lastline = 0
            for linenum, line in enumerate(fh.readlines(), 1):
                lastline = linenum
                if linenum % 4 == 1:
                    read_id = line.strip()
                    matches = self.id_regex.match(read_id)
                    assert matches, BAD_ID % (linenum, read_id, self.id_regex)
                    frag_start, frag_end = [
                        int(c) for c in matches.groups()[1:3]]
                    read = int(matches.groups()[-1])

                elif linenum % 4 == 2:
                    seq = line.strip()
                    if self.read_length == -1:
                        self.read_length = len(seq)
                    else:
                        assert len(seq) == self.read_length, \
                            BAD_SEQLEN % (linenum, seq, self.read_length)
                    disallowed = set(seq) - self.allowed_bases
                    assert not disallowed, \
                        BAD_BASES % (linenum, disallowed, self.allowed_bases)

                elif linenum % 4 == 3:
                    plus = line.strip()
                    assert plus == "+", BAD_PLUS % (linenum, plus)

                if linenum % 4 == 0:
                    quals = line.strip()
                    assert len(quals) == self.read_length, \
                        BAD_QUALS % (linenum, quals, self.read_length)

                    self.records.append({
                        "id": read_id, "seq": seq, "quals": quals,
                        "frag_start": frag_start, "frag_end": frag_end,
                        "read": read})
                    low = min(frag_start, frag_end)
                    high = max(frag_start, frag_end)
                    if self.minpos == -1 or low < self.minpos:
                        self.minpos = low
                    if self.maxpos == -1 or high > self.maxpos:
                        self.maxpos = high

            assert lastline % 4 == 0, MSG_INCOMPLETE % (filename)

    def coverage(self, pos):
        """Return reads covering pos"""
        # simple logic if all reads are forward on the reference strand:
        if self.forwardized:
            return [r for r in self.records if
                    r['read_start'] <= pos <=
                    r['read_start'] + self.read_length - 1]
        # more cases to consider if not:
        else:
            covering = []
            for r in self.records:
                start = min(r['frag_start'], r['frag_end'])
                end = max(r['frag_start'], r['frag_end'])
                read = r['read']
                flipped = True if r['frag_start'] > r['frag_end'] else False
                if (read == 1 and not flipped and
                        start <= pos <= start + self.read_length - 1 or
                    read == 2 and not flipped and
                        end - self.read_length + 1 <= pos <= end or
                    read == 1 and flipped and
                        end - self.read_length + 1 <= pos <= end or
                    read == 2 and flipped and
                        start <= pos <= start + self.read_length - 1):
                    covering.append(r)
            return covering

    def basecounts(self):
        """Return a dict of { base: count } aggregated over all reads"""
        counts = {}
        for r in self.records:
            for base in r['seq']:
                counts[base] = counts.setdefault(base, 0) + 1
        return counts

    @classmethod
    def forwardize(cls, original):
        """Return a copy of original with all reads turned into forward reads:
           a calculational convenience"""
        fwdized = copy.deepcopy(original)
        for r in fwdized.records:
            frag_start, frag_end = r['frag_start'], r['frag_end']
            read = r['read']
            if (read == 1 and frag_start < frag_end):
                r['read_start'] = frag_start
            elif (read == 1 and frag_start > frag_end):
                r['seq'] = ''.join(cls.revcomp(r['seq']))
                r['quals'] = ''.join(reversed(r['quals']))
                r['read_start'] = frag_start - fwdized.read_length + 1
            elif (read == 2 and frag_start < frag_end):
                r['seq'] = ''.join(cls.revcomp(r['seq']))
                r['quals'] = ''.join(reversed(r['quals']))
                r['read_start'] = frag_end - fwdized.read_length + 1
            elif (read == 2 and frag_start > frag_end):
                r['read_start'] = frag_end
            else:
                raise Exception("Unhandled case:", r)
        fwdized.forwardized = True
        return fwdized

    @classmethod
    def revcomp(cls, seq):
        return [cls.complement[b] for b in reversed(seq)]
