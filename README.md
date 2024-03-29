# qasim
Simulate mutations, fragment generation and paired-end HTS reads on diploid[1] genomes.

This project started life as a fork of Heng Li's [wgsim](https://github.com/lh3/wgsim) tool, which has been fairly heavily revised as a Python C-extension module with unit tests and extra functionality.

The principal additional features are:
1. The ability to generate realistic base qualities when provided with appropriate template files: see `tests/resources/R1.qp.xml` for an example. Suitable inputs for this purpose may be generated by running [qProfiler](https://github.com/AdamaJava/adamajava/tree/master/qprofiler) on real-world sequencing data.
2. `VCF` is used as both input and output format: you can specify an exact set of mutations instead of random mutations by supplying an input VCF file, and when simulating random mutations they are recorded in an output VCF file.
3. `-S, --somatic-mode` in which the mutation and read generation cycle is run twice; "somatic" mutations being applied on top of "germline" mutations. The simulated reads are then drawn from a mixture of the two pools in a ratio specified by the user, to represent for example tumour sample (im)purity.
4. It's a Python module so you can just ```import qasim.qasim as qq``` but the guts of it is written in C & Cython so it's fast.


# build
```bash
# clean
./setup.py clean

# build locally (requires Cython)
./setup.py build_ext --inplace

# test
./setup.py test

# install
./setup.py install
```

# help

<pre>
qasim_cli.py -h

usage: qasim_cli.py [-h] [-r {Range 0.0<=<1.0}] [-H {Range 0.0<=<1.0}] [-R {Range 0.0<=<1.0}] [-X {Range 0.0<=<1.0}]
                    [-M MAX_INSERTION] [-n SAMPLE_NAME] (-o VCF | -V VCF_INPUT) [-S] [--mut-rate2 {Range 0.0<=<1.0}]
                    [--homo-frac2 {Range 0.0<=<1.0}] [--indel-frac2 {Range 0.0<=<1.0}] [--indel-extend2 {Range 0.0<=<1.0}]
                    [--max-insertion2 MAX_INSERTION2] [--contamination {Range 0.0<=<1.0}] [--sample-name2 SAMPLE_NAME2]
                    [--output2 VCF2 | --vcf-input2 VCF_INPUT2] [-z SIZE] [-s STD_DEV] [--AC {Range 0.0<=<1.0}]
                    [--AG {Range 0.0<=<1.0}] [--AT {Range 0.0<=<1.0}] [--CA {Range 0.0<=<1.0}] [--CG {Range 0.0<=<1.0}]
                    [--CT {Range 0.0<=<1.0}] [--GA {Range 0.0<=<1.0}] [--GC {Range 0.0<=<1.0}] [--GT {Range 0.0<=<1.0}]
                    [--TA {Range 0.0<=<1.0}] [--TC {Range 0.0<=<1.0}] [--TG {Range 0.0<=<1.0}] [-N NUM_PAIRS] [-1 LENGTH1]
                    [-2 LENGTH2] [-A {Range 0.0<=<1.0}] [-e {Range 0.0<=<1.0} | -Q R1_QUALS R2_QUALS]
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
                        fraction of mutations that are homozygous (default: 0.333333)
  -R {Range 0.0<=<1.0}, --indel-frac {Range 0.0<=<1.0}
                        fraction of mutations that are indels (default: 0.15)
  -X {Range 0.0<=<1.0}, --indel-extend {Range 0.0<=<1.0}
                        probability an indel is extended (default: 0.3)
  -M MAX_INSERTION, --max-insertion MAX_INSERTION
                        Maximum size of generated insertions (regardless of -X value) (default: 1000)
  -n SAMPLE_NAME, --sample-name SAMPLE_NAME
                        name of sample for vcf output (default: SAMPLE)
  -o VCF, --output VCF  output generated mutations to file (default: None)
  -V VCF_INPUT, --vcf-input VCF_INPUT
                        use input vcf file as source of mutations instead of randomly generating them. VCF records must be
                        grouped by CHROM and ordered by POS within each CHROM. (default: None)

Somatic mutations:
  If "-S, --somatic-mode" is specified then mutation and read generation will be run /twice/ - the first time generating
  "germline" mutations, the second time generating "somatic" mutations. Specifying the other options in this group has no
  effect if not in somatic mode.

  -S, --somatic-mode
  --mut-rate2 {Range 0.0<=<1.0}
                        somatic mutation rate (default: 1e-06)
  --homo-frac2 {Range 0.0<=<1.0}
                        fraction of somatic mutations that are homozygous (default: 0.333333)
  --indel-frac2 {Range 0.0<=<1.0}
                        fraction of somatic mutations that are indels (default: 0.15)
  --indel-extend2 {Range 0.0<=<1.0}
                        probability a somatic indel is extended (default: 0.3)
  --max-insertion2 MAX_INSERTION2
                        Maximum size of generated somatic insertions (regardless of -X value) (default: 1000)
  --contamination {Range 0.0<=<1.0}
                        fraction of reads generated from "germline" sequence (default: 0.0)
  --sample-name2 SAMPLE_NAME2
                        name of sample for vcf2 output (default: SOMATIC)
  --output2 VCF2        output generated somatic mutations to file (default: None)
  --vcf-input2 VCF_INPUT2
                        use input vcf file as source of somatic mutations instead of randomly generating them. VCF records
                        must be grouped by CHROM and ordered by POS within each CHROM. (default: None)

Fragments:
  The transition/transversion rates represent the chance that the given random base conversion occurs at any position.
  This is applied after fragment generation but before sequencing read error, and can be used to model sample degradation,
  e.g. with a non-zero C>T rate for FFPE samples.

  -z SIZE, --size SIZE  mean fragment size (default: 500)
  -s STD_DEV, --std-dev STD_DEV
                        fragment standard deviation (default: 50)
  --AC {Range 0.0<=<1.0}
                        A>C transversion rate (default: None)
  --AG {Range 0.0<=<1.0}
                        A>G transition rate (default: None)
  --AT {Range 0.0<=<1.0}
                        A>T transversion rate (default: None)
  --CA {Range 0.0<=<1.0}
                        C>A transversion rate (default: None)
  --CG {Range 0.0<=<1.0}
                        C>G transversion rate (default: None)
  --CT {Range 0.0<=<1.0}
                        C>T transition rate (default: None)
  --GA {Range 0.0<=<1.0}
                        G>A transition rate (default: None)
  --GC {Range 0.0<=<1.0}
                        G>C transversion rate (default: None)
  --GT {Range 0.0<=<1.0}
                        G>T transversion rate (default: None)
  --TA {Range 0.0<=<1.0}
                        T>A transversion rate (default: None)
  --TC {Range 0.0<=<1.0}
                        T>C transition rate (default: None)
  --TG {Range 0.0<=<1.0}
                        T>G transversion rate (default: None)

Reads:
  If -e is specified then a fixed error rate (and quality string) is used along the entire read. If -Q is specified then
  quality scores will be randomly generated according to the distributions specified in the two files, and the error rate
  and quality value is calculated per base. The files specified by -Q should be qprofiler-like XML documents with &lt;QUAL&gt;
  elements containing &lt;Cycle&gt; and &lt;TallyItem&gt; elements with "count" and "value" attributes.

  -N NUM_PAIRS, --num-pairs NUM_PAIRS
                        number of read pairs (default: 1000000)
  -1 LENGTH1, --length1 LENGTH1
                        length of read 1 (default: 100)
  -2 LENGTH2, --length2 LENGTH2
                        length of read 2 (default: 100)
  -A {Range 0.0<=<1.0}, --ambig-frac {Range 0.0<=<1.0}
                        discard read if fraction of "N" bases exceeds this (default: 1.0)
  -e {Range 0.0<=<1.0}, --error-rate {Range 0.0<=<1.0}
                        read error rate (constant) (default: 0.002)
  -Q R1_QUALS R2_QUALS, --quals-from R1_QUALS R2_QUALS
                        generate random quality strings for read 1 and read 2 respectively from the distributions specified
                        in the files (default: None)
  --num-quals NUM_QUALS
                        number of quality strings to generate from distribution files (default: 10000)

Other:
  -d SEED, --seed SEED  seed for random generator (default=current time)
  -t, --test-output     print mutated sequences to stdout (default: False)
  -w, --wgsim-mode      In this mode insertions are generated using the same logic as original wgsim.c - i.e. max_insertion
                        is set to 4, and insert bases are reversed with respect to generation order. (default: False)

</pre>

***

[1] For the purposes of sequence and fragment generation all contigs of the supplied reference are treated on equal footing
    as representing   diploid chromosomes - in particular there is no special treatment of the different relative abundance of `MT` or the human sex allosomes. This means, for example, if it is important to generate approximately realistic relative coverage of the autosomes, `chrX` and `chrY` for a human male, separate simulations should be combined from a reference containing `chr1` - `chr22`, and another containing `chrX` and `chrY` with `--num-pairs` adjusted proportionally.
