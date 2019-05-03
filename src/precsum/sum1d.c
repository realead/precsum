#define BLOCKSIZE 128

//similar to pairwise implementation of numpy's pairwise_add

typedef unsigned int index_t;
typedef float value_t;
value_t pairwise_1dsum_FLOAT(const value_t *a, index_t n, index_t stride)
{
    if (n < 8) {
        index_t i;
        value_t res = 0.;
        for (i = 0; i < n; i++) {
            res += a[i * stride];
        }
        return res;
    }
    else if (n <= BLOCKSIZE) {
        index_t i;
        value_t r[8], res;

        /*
         * sum a block with 8 accumulators
         * 8 times unroll reduces blocksize to 16 and allows vectorization with
         * avx without changing summation ordering
         */
        r[0] = (a[0 * stride]);
        r[1] = (a[1 * stride]);
        r[2] = (a[2 * stride]);
        r[3] = (a[3 * stride]);
        r[4] = (a[4 * stride]);
        r[5] = (a[5 * stride]);
        r[6] = (a[6 * stride]);
        r[7] = (a[7 * stride]);

        for (i = 8; i < n - (n % 8); i += 8) {
            r[0] += (a[(i + 0) * stride]);
            r[1] += (a[(i + 1) * stride]);
            r[2] += (a[(i + 2) * stride]);
            r[3] += (a[(i + 3) * stride]);
            r[4] += (a[(i + 4) * stride]);
            r[5] += (a[(i + 5) * stride]);
            r[6] += (a[(i + 6) * stride]);
            r[7] += (a[(i + 7) * stride]);
        }

        /* accumulate now to avoid stack spills for single peel loop */
        res = ((r[0] + r[1]) + (r[2] + r[3])) +
              ((r[4] + r[5]) + (r[6] + r[7]));

        /* do non multiple of 8 rest */
        for (; i < n; i++) {
            res += (a[i * stride]);
        }
        return res;
    }
    else {
        /* divide by two but avoid non-multiples of unroll factor */
        index_t n2 = n / 2;
        n2 -= n2 % 8;
        return pairwise_1dsum_FLOAT(a, n2, stride) +
               pairwise_1dsum_FLOAT(a + n2 * stride, n - n2, stride);
    }
}

