/* ----------------------------------------------------------------- */
/*             The Speech Signal Processing Toolkit (SPTK)           */
/*             developed by SPTK Working Group                       */
/*             http://sp-tk.sourceforge.net/                         */
/* ----------------------------------------------------------------- */
/*                                                                   */
/*  Copyright (c) 1984-2007  Tokyo Institute of Technology           */
/*                           Interdisciplinary Graduate School of    */
/*                           Science and Engineering                 */
/*                                                                   */
/*                1996-2015  Nagoya Institute of Technology          */
/*                           Department of Computer Science          */
/*                                                                   */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/* - Redistributions of source code must retain the above copyright  */
/*   notice, this list of conditions and the following disclaimer.   */
/* - Redistributions in binary form must reproduce the above         */
/*   copyright notice, this list of conditions and the following     */
/*   disclaimer in the documentation and/or other materials provided */
/*   with the distribution.                                          */
/* - Neither the name of the SPTK working group nor the names of its */
/*   contributors may be used to endorse or promote products derived */
/*   from this software without specific prior written permission.   */
/*                                                                   */
/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND            */
/* CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,       */
/* INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF          */
/* MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE          */
/* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS */
/* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,          */
/* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED   */
/* TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,     */
/* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON */
/* ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,   */
/* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY    */
/* OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           */
/* POSSIBILITY OF SUCH DAMAGE.                                       */
/* ----------------------------------------------------------------- */

/*************************************************************************
 *                                                                       *
 *    Calculation of GMM log-probagility                                 *
 *                                                                       *
 *                                       2000.7  C. Miyajima             *
 *                                                                       *
 *       usage:                                                          *
 *               gmmp [options] gmmfile [infile] > stdout                *
 *       options:                                                        *
 *               -l l  :  length of vector                      [26]     *
 *               -m m  :  number of Gaussian components         [16]     *
 *               -f    :  full covariance                       [FALSE]  *
 *               -a    :  output average log-probability        [FALSE]  *
 *             (level 2)                                                 *
 *               -B B1 ... Bb : block size in covariance matrix [FALSE]  *
 *                              where (B1 + B2 + ... + Bb) = l           *
 *               -c1   : inter-block correlation                [FALSE]  *
 *               -c2   : full covariance in each block          [FALSE]  *
 *       infile:                                                         *
 *               input vector sequence                          [stdin]  *
 *       stdout:                                                         *
 *               sequence of frame log-probabilities                     *
 *               average log-probability (if -a is used)                 *
 *                                                                       *
 ************************************************************************/

static char *rcs_id = "$Id: gmmp.c,v 1.21 2015/12/14 05:34:34 uratec Exp $";

/*  Standard C Libraries  */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#ifdef HAVE_STRING_H
#include <string.h>
#else
#include <strings.h>
#ifndef HAVE_STRRCHR
#define strrchr rindex
#endif
#endif

#include <math.h>

#if defined(WIN32)
#include "SPTK.h"
#else
#include <SPTK.h>
#endif

/*  Default Values  */

#define DEF_L  26
#define DEF_M  16
#define DEF_A  FA
#define FULL   FA

char *BOOL[] = { "FALSE", "TRUE" };

/*  Command Name  */
char *cmnd;

void usage(int status)
{
   fprintf(stderr, "\n");
   fprintf(stderr, " %s - Calculaton of GMM log-probability\n", cmnd);
   fprintf(stderr, "\n");
   fprintf(stderr, "  usage:\n");
   fprintf(stderr, "       %s [ options ] gmmfile [ infile ] > stdout\n", cmnd);
   fprintf(stderr, "  options:\n");

   fprintf(stderr,
           "       -l l  : dimensionality of vectors                   [%d]\n",
           DEF_L);
   fprintf(stderr,
           "       -m m  : number of Gaussian components               [%d]\n",
           DEF_M);
   fprintf(stderr,
           "       -f    : full covariance                             [%s]\n",
           BOOL[FULL]);
   fprintf(stderr,
           "       -a    : output average log-probability              [%s]\n",
           BOOL[DEF_A]);
   fprintf(stderr, "       -h    : print this message\n");
   fprintf(stderr, "     (level 2)\n");
   fprintf(stderr,
           "       -B B1 B2 ... Bb : block size in covariance matrix,  [FALSE]\n");
   fprintf(stderr, "                         where (B1 + B2 + ... + Bb) = l\n");
   fprintf(stderr,
           "       -c1   : inter-block correlation                     [FALSE]\n");
   fprintf(stderr,
           "       -c2   : full covariance in each block               [FALSE]\n");
   fprintf(stderr, "  infile:\n");
   fprintf(stderr,
           "       input data sequence (float)                         [stdin]\n");
   fprintf(stderr, "  gmmfile:\n");
   fprintf(stderr, "       GMM parameters (float)\n");
   fprintf(stderr, "  stdout:\n");
   fprintf(stderr,
           "       log-probabilities or average log-probability (float)\n");
   fprintf(stderr, "  notice:\n");
   fprintf(stderr,
           "       -B option specifies the size of each blocks in covariance matrix.\n");
   fprintf(stderr, "       -c1 and -c2 option must be used with -B option.\n");
   fprintf(stderr,
           "         Without -c1 and -c2 option, a diagonal covariance can be obtained.\n");
#ifdef PACKAGE_VERSION
   fprintf(stderr, "\n");
   fprintf(stderr, " SPTK: version %s\n", PACKAGE_VERSION);
   fprintf(stderr, " CVS Info: %s", rcs_id);
#endif
   fprintf(stderr, "\n");
   exit(status);
}

int main(int argc, char **argv)
{
   FILE *fp = stdin, *fgmm = NULL;
   GMM gmm;
   double logp, ave_logp, *x;
   int M = DEF_M, L = DEF_L, T;
   Boolean aflag = DEF_A, full = FULL;
   int cov_dim = 0, dim_list[1024], i, j;
   Boolean block_full = FA, block_corr = FA, multiple_dim = FA;

   if ((cmnd = strrchr(argv[0], '/')) == NULL)
      cmnd = argv[0];
   else
      cmnd++;


   while (--argc)
      if (**++argv == '-') {
         switch (*(*argv + 1)) {
         case 'h':
            usage(0);
            break;
         case 'l':
            L = atoi(*++argv);
            --argc;
            break;
         case 'm':
            M = atoi(*++argv);
            --argc;
            break;
         case 'f':
            full = TR - full;
            break;
         case 'a':
            aflag = TR;
            break;
            /* level 2 */
         case 'B':
            multiple_dim = TR;
            dim_list[0] = atoi(*++argv);
            i = 1;
            argc--;
            while ((**(argv + 1) != '\0') && isdigit(**(argv + 1))) {
               dim_list[i++] = atoi(*++argv);
               if (--argc <= 1) {
                  break;
               }
            }
            cov_dim = i;
            break;
         case 'c':
            if (strncmp("1", *(argv) + 2, 1) == 0) {
               block_corr = TR - block_corr;
            } else if (strncmp("2", *(argv) + 2, 1) == 0) {
               block_full = TR - block_full;
            }
            break;
         default:
            fprintf(stderr, "%s: Illegal option \"%s\".\n", cmnd, *argv);
            usage(1);
         }
      } else if (fgmm == NULL)
         fgmm = getfp(*argv, "rb");
      else
         fp = getfp(*argv, "rb");

   if (multiple_dim == TR) {
      for (i = 0, j = 0; i < cov_dim; i++) {
         j += dim_list[i];
      }
      if (j != L) {
         fprintf(stderr,
                 "%s: block size must be coincided with dimention of vector\n",
                 cmnd);
         usage(1);
      }
      full = TR;
   } else {
      if (block_corr == TR || block_full == TR) {
         if (full != TR) {
            fprintf(stderr,
                    "%s: -c1 and -c2 option must be specified with -B option!\n",
                    cmnd);
            usage(1);
         }
      }
   }

   /* Read GMM parameters */
   if (fgmm == NULL) {
      fprintf(stderr, "%s: GMM file must be specified!\n", cmnd);
      usage(1);
   }

   alloc_GMM(&gmm, M, L, full);
   load_GMM(&gmm, fgmm);

   fclose(fgmm);

   if (gmm.full != TR && full == TR) {
      fprintf(stderr,
              "%s: GMM is diagonal covariance, so -f or -B option is inappropriate!\n",
              cmnd);
      usage(1);
   }

   if (full == TR && multiple_dim == TR) {
      maskCov_GMM(&gmm, dim_list, cov_dim, block_full, block_corr);
   }

   if (gmm.full == TR) {
      prepareCovInv_GMM(&gmm);
      prepareGconst_GMM(&gmm);
   }

   /* Calculate and output log-probability */
   T = 0;
   ave_logp = 0.0;
   x = dgetmem(L);
   while (freadf(x, sizeof(*x), L, fp) == L) {
      if (!aflag) {
         logp = log_outp(&gmm, L, x);
         fwritef(&logp, sizeof(double), 1, stdout);
      } else {
         ave_logp += log_outp(&gmm, L, x);
         T++;
      }
   }
   fclose(fp);

   if (aflag) {
      if (T == 0) {
         fprintf(stderr, "%s: No input data!\n", cmnd);
         usage(1);
      } else {
         ave_logp /= (double) T;
         fwritef(&ave_logp, sizeof(double), 1, stdout);
      }
   }

   return (0);
}
