# qasim
Simulate mutations, fragment generation and shotgun sequencing on genomes.

# build
```bash
# build locally (requires Cython)
./setup.py build_ext --inplace

# install
./setup.py install

# clean
./setup.py clean

# create distribution
./setup.py sdist
```

# help

```bash
./qasim_wrapper.py -h

usage: qasim_wrapper.py [-h] [-r {Range 0.0<=<1.0}] [-H {Range 0.0<=<1.0}]
                        [-R {Range 0.0<=<1.0}] [-X {Range 0.0<=<1.0}]
                        [-M MAX_INSERTION] [-n SAMPLE_NAME]
                        (-o VCF | -V VCF_INPUT) [-S]
                        [--mut-rate2 {Range 0.0<=<1.0}]
                        [--homo-frac2 {Range 0.0<=<1.0}]
                        [--indel-frac2 {Range 0.0<=<1.0}]
                        [--indel-extend2 {Range 0.0<=<1.0}]
                        [--max-insertion2 MAX_INSERTION2]
                        [--contamination {Range 0.0<=<1.0}]
                        [--sample-name2 SAMPLE_NAME2]
                        [--output2 VCF2 | --vcf-input2 VCF_INPUT2] [-z SIZE]
                        [-s STD_DEV] [-N NUM_PAIRS] [-1 LENGTH1] [-2 LENGTH2]
                        [-A {Range 0.0<=<1.0}]
                        [-e {Range 0.0<=<1.0} | -Q QUALS_FROM]
                        [--num-quals NUM_QUALS] [-d SEED] [-t] [-w]
                        fasta read1fq read2fq

Simulate mutations, fragment generation and shotgun sequencing on genomes.

positional arguments:
  fasta                 Reference FASTA
  read1fq               Output file for read1
  read2fq               Output file for read2

optional arguments:
  -h, --help            show this help message and exit

Mutations:
  -r {Range 0.0<=<1.0}, --mut-rate {Range 0.0<=<1.0}
                        mutation rate (default: 0.001)
  -H {Range 0.0<=<1.0}, --homo-frac {Range 0.0<=<1.0}
                        fraction of mutations that are homozygous (default:
                        0.333333)
  -R {Range 0.0<=<1.0}, --indel-frac {Range 0.0<=<1.0}
                        fraction of mutations that are indels (default: 0.15)
  -X {Range 0.0<=<1.0}, --indel-extend {Range 0.0<=<1.0}
                        probability an indel is extended (default: 0.3)
  -M MAX_INSERTION, --max-insertion MAX_INSERTION
                        Maximum size of generated insertions (regardless of -X
                        value) (default: 1000)
  -n SAMPLE_NAME, --sample-name SAMPLE_NAME
                        name of sample for vcf output (default: SAMPLE)
  -o VCF, --output VCF  output generated mutations to file (default: None)
  -V VCF_INPUT, --vcf-input VCF_INPUT
                        use input vcf file as source of mutations instead of
                        randomly generating them (default: None)

Somatic mutations:
  If "-S, --somatic-mode" is specified then mutation and read generation
  will be run /twice/ - the first time generating "germline" mutations, the
  second time generating "somatic" mutations. Specifying the other options
  in this group has no effect if not in somatic mode.

  -S, --somatic-mode
  --mut-rate2 {Range 0.0<=<1.0}
                        somatic mutation rate (default: 1e-06)
  --homo-frac2 {Range 0.0<=<1.0}
                        fraction of somatic mutations that are homozygous
                        (default: 0.333333)
  --indel-frac2 {Range 0.0<=<1.0}
                        fraction of somatic mutations that are indels
                        (default: 0.15)
  --indel-extend2 {Range 0.0<=<1.0}
                        probability a somatic indel is extended (default: 0.3)
  --max-insertion2 MAX_INSERTION2
                        Maximum size of generated somatic insertions
                        (regardless of -X value) (default: 1000)
  --contamination {Range 0.0<=<1.0}
                        fraction of reads generated from "germline" sequence
                        (default: 0.0)
  --sample-name2 SAMPLE_NAME2
                        name of sample for vcf2 output (default: SOMATIC)
  --output2 VCF2        output generated somatic mutations to file (default:
                        None)
  --vcf-input2 VCF_INPUT2
                        use input vcf file as source of somatic mutations
                        instead of randomly generating them (default: None)

Fragments:
  -z SIZE, --size SIZE  mean fragment size (default: 500)
  -s STD_DEV, --std-dev STD_DEV
                        fragment standard deviation (default: 50)

Reads:
  If -e is specified then a fixed error rate (and quality string) is used
  along the entire read. If -Q is specified then quality scores will be
  randomly generated according to the distribution specified in the file,
  and the error rate and quality value is calculated per base. The file
  specified by -Q should be a qprofiler-like XML document with a <QUAL>
  element containing <Cycle> and <TallyItem> elements with "count" and
  "value" attributes. If -Q is specified, read1 and read2 lengths must be
  the same.

  -N NUM_PAIRS, --num-pairs NUM_PAIRS
                        number of read pairs (default: 1000000)
  -1 LENGTH1, --length1 LENGTH1
                        length of read 1 (default: 100)
  -2 LENGTH2, --length2 LENGTH2
                        length of read 2 (default: 100)
  -A {Range 0.0<=<1.0}, --ambig-frac {Range 0.0<=<1.0}
                        discard read if fraction of "N" bases exceeds this
                        (default: 0.05)
  -e {Range 0.0<=<1.0}, --error-rate {Range 0.0<=<1.0}
                        read error rate (constant) (default: 0.002)
  -Q QUALS_FROM, --quals-from QUALS_FROM
                        generate random quality strings from the distribution
                        specified in file (default: None)
  --num-quals NUM_QUALS
                        number of quality strings to generate from
                        distribution file (default: 1000)

Other:
  -d SEED, --seed SEED  seed for random generator (default=current time)
                        (default: 1543554349)
  -t, --test-output     print mutated sequences to stdout (default: False)
  -w, --wgsim-mode      In this mode insertions are generated using the same
                        logic as original wgsim.c - i.e. max_insertion is set
                        to 4, and insert bases are reversed with respect to
                        generation order. (default: False)

```


