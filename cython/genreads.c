/* The MIT License

   cython/genreads.c is a modified version of the core loop of wgsim.c
   (https://github.com/lh3/wgsim):
   Copyright (c) 2008 Genome Research Ltd (GRL).
                 2011 Heng Li <lh3@live.co.uk>
   all other code:
   Copyright (c) 2013,2018 Conrad Leonard <conrad.leonard@hotmail.com>

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/*
 * Changes 2013,2018 Conrad Leonard <conrad.leonard@hotmail.com>
 * This code intended to be called from Cython extension qasim.
 *
 * genreads() largely follows wgsim.c (https://github.com/lh3/wgsim)
 * but using actual base representation of mutated sequence, rather than
 * mutmsk and bitshifting to represent indels. This is much simpler and 
 * correctly generates reverse reads over insertions. Additionally we allow
 * here per-base qualities and sequencing error from per-base Phred score.
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <ctype.h>
#include <string.h>


/* Simple normal random number generator, copied from genran.c */

double rand_normal(void)
{ 
    static int iset = 0; 
    static double gset; 
    double fac, rsq, v1, v2; 
    if (iset == 0) {
        do { 
            v1 = 2.0 * drand48() - 1.0;
            v2 = 2.0 * drand48() - 1.0; 
            rsq = v1 * v1 + v2 * v2;
        } while (rsq >= 1.0 || rsq == 0.0);
        fac = sqrt(-2.0 * log(rsq) / rsq); 
        gset = v1 * fac; 
        iset = 1;
        return v2 * fac;
    } else {
        iset = 0;
        return gset;
    }
}


int genreads(FILE *fpout1, FILE *fpout2, uint8_t *s1, uint8_t *s2,
            uint32_t *rel1, uint32_t *rel2, uint32_t len1, uint32_t len2,
            uint64_t n_pairs, int dist, int std_dev, int size_l, int size_r, 
            double ERR_RATE, double MAX_N_RATIO, const char *seqname,
            int num_quals, double **p, char **q)
{
    uint8_t *rseq[2] = { s1, s2 };      // base sequence
    uint32_t *rel[2] = { rel1, rel2 };  // reference-relative positions
    uint64_t ii;
    int i, size[2], Q, max_size;
    char *qstr[2];
    double *pvals[2];
    uint8_t *tmp_seq[2];
    uint8_t *target;
    uint32_t *target_rel;
    uint32_t len[2] = { len1, len2 }, target_len;

    max_size = size_l > size_r? size_l : size_r;
    qstr[0] = (char*)calloc(max_size+1, 1);
    qstr[1] = (char*)calloc(max_size+1, 1);
    pvals[0] = (double*)calloc(max_size+1, sizeof(double));
    pvals[1] = (double*)calloc(max_size+1, sizeof(double));
    tmp_seq[0] = (uint8_t*)calloc(max_size+2, 1);
    tmp_seq[1] = (uint8_t*)calloc(max_size+2, 1);
    size[0] = size_l; size[1] = size_r;

    // 'I' corresponds to a QUAL of 40 => ERR_RATE = 0.0001
    Q = (ERR_RATE == 0.0)? 'I' : (int)(-10.0 * log(ERR_RATE) / log(10.0) + 0.499) + 33;

    for (ii = 0; ii != n_pairs; ++ii) { 
        double ransz, ranerr;
        int d, pos, s[2], is_flip = 0, ran01, ranq;
        int n_err[2], ext_coor[2], j, k;
        FILE *fpo[2];

        // generate the read sequences
        ran01 = drand48()<0.5?0:1; // haplotype from which the reads are generated
        target = rseq[ran01];
        target_rel = rel[ran01];
        target_len = len[ran01]; 
        n_err[0] = n_err[1] = 0;

        do { // avoid boundary failure
            ransz = rand_normal();
            ransz = ransz * std_dev + dist;
            d = (int)(ransz + 0.5);
            // ensure frag length > read length
            d = d > max_size? d : max_size;
            pos = (int)((target_len - d + 1) * drand48());
        } while (pos < 0 || pos + d - 1 >= target_len);

        // flip or not
        if (drand48() < 0.5) {
            fpo[0] = fpout1; fpo[1] = fpout2;
            s[0] = size[0]; s[1] = size[1];
        } else {
            fpo[1] = fpout1; fpo[0] = fpout2;
            s[1] = size[0]; s[0] = size[1];
            is_flip = 1;
        }

        // read 0
        ext_coor[0] = target_rel[pos];
        for (i = pos, k = 0; k < s[0]; i++) { 
            tmp_seq[0][k++] = target[i];
        }

        // read 1
        ext_coor[1] = target_rel[pos + d -  1];
        for (i = pos + d - 1, k = 0; k < s[1]; i--) {    
            tmp_seq[1][k++] = target[i];
        }

        for (k = 0; k < s[1]; ++k) tmp_seq[1][k] = tmp_seq[1][k] < 4? 3 - tmp_seq[1][k] : 4; // complement

        // make quality strings
        for (j = 0; j < 2; j++) {
            if (num_quals) {
                ranq = (int)(drand48() * num_quals);
                for (i = 0; i < s[j]; ++i) {
                    qstr[j][i] = q[ranq][i] + 33;                        // qual string from distribution
                    pvals[j][i] = p[ranq][i];
                }
            } else {
                for (i = 0; i < s[j]; ++i) qstr[j][i] = Q;               // qual string from fixed ERR_RATE
            } 
        }

        // generate sequencing errors
        for (j = 0; j < 2; ++j) {
            int n_n = 0;
            for (i = 0; i < s[j]; ++i) {
                int c = tmp_seq[j][i];
                ranerr = drand48();
                if (c >= 4) { 
                    c = 4;
                    ++n_n;
                } else if ((num_quals && (ranerr < pvals[j][i])) ||   // error from qual score
                          (!num_quals && (ranerr < ERR_RATE))){       // error from fixed rate
                    c = (c + (int)(drand48() * 3.0 + 1)) & 3;         // random sequencing errors
                    //c = (c + 1) & 3;                                // recurrent sequencing errors
                    ++n_err[j];
                }
                tmp_seq[j][i] = c;
            }
            if ((double)n_n / s[j] > MAX_N_RATIO) break;
        }
        if (j < 2) { // too many ambiguous bases on one of the reads
            --ii;
            continue;
        }

        // print
        for (j = 0; j < 2; ++j) {
            fprintf(fpo[j], "@%s_%u_%u_e%d_e%d_%llx/%d\n", seqname,
                    is_flip==0? ext_coor[0] : ext_coor[1], is_flip==0? ext_coor[1] : ext_coor[0],
                    n_err[0], n_err[1], (long long)ii, j==0? is_flip+1 : 2-is_flip);
            for (i = 0; i < s[j]; ++i)
                fputc("ACGTN"[(int)tmp_seq[j][i]], fpo[j]);
            fprintf(fpo[j], "\n+\n%s\n", qstr[j]);
        }
    }
    fflush(fpout1); fflush(fpout2);
    free(qstr[0]); free(qstr[1]);
    free(pvals[0]); free(pvals[1]);
    free(tmp_seq[0]); free(tmp_seq[1]);
    return 0;
}
