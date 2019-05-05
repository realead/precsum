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


#define ROW_COUNT 16
void pairwise_blocksum_FLOAT(const value_t *a, index_t n, index_t stride_along, index_t m, index_t stride_crosswise, value_t *output, index_t stride_output)
{
    //we trust m to be<=ROW_COUNT
    if (n < 8) {
        index_t i,j;
        for( j = 0; j < m; j++){
               output[j*stride_output]=a[j*stride_crosswise];
            }
        for (i = 1; i < n; i++) { 
           for( j = 0; j < m; j++){          
               output[j*stride_output] += a[i * stride_along + j*stride_crosswise];
           }
        }
    }
    else if (n <= BLOCKSIZE) {
        index_t i,j;
        value_t r[ROW_COUNT][8];

        for(j = 0; j<m; j++){
            r[j][0] = (a[0 * stride_along + j*stride_crosswise]);
            r[j][1] = (a[1 * stride_along + j*stride_crosswise]);
            r[j][2] = (a[2 * stride_along + j*stride_crosswise]);
            r[j][3] = (a[3 * stride_along + j*stride_crosswise]);
            r[j][4] = (a[4 * stride_along + j*stride_crosswise]);
            r[j][5] = (a[5 * stride_along + j*stride_crosswise]);
            r[j][6] = (a[6 * stride_along + j*stride_crosswise]);
            r[j][7] = (a[7 * stride_along + j*stride_crosswise]);
        }
        for (i = 8; i < n - (n % 8); i += 8) {
          for(j = 0; j<m; j++){
            r[j][0] += (a[(i + 0) * stride_along + j*stride_crosswise]);
            r[j][1] += (a[(i + 1) * stride_along + j*stride_crosswise]);
            r[j][2] += (a[(i + 2) * stride_along + j*stride_crosswise]);
            r[j][3] += (a[(i + 3) * stride_along + j*stride_crosswise]);
            r[j][4] += (a[(i + 4) * stride_along + j*stride_crosswise]);
            r[j][5] += (a[(i + 5) * stride_along + j*stride_crosswise]);
            r[j][6] += (a[(i + 6) * stride_along + j*stride_crosswise]);
            r[j][7] += (a[(i + 7) * stride_along + j*stride_crosswise]);
          }
        }


       for( j = 0; j < m; j++){          
           output[j*stride_output] = ((r[j][0] + r[j][1]) + (r[j][2] + r[j][3])) +
                       ((r[j][4] + r[j][5]) + (r[j][6] + r[j][7]));
       }

        /* do non multiple of 8 rest */
       for (; i < n; i++) {
            for(j = 0; j<m; j++){
                output[j*stride_output] += (a[i * stride_along + j*stride_crosswise]);
            }
       }
    }
    else {
        /* divide by two but avoid non-multiples of unroll factor */
        index_t n2 = n / 2;
        n2 -= n2 % 8;
        value_t first[ROW_COUNT];
        value_t second[ROW_COUNT];
        pairwise_blocksum_FLOAT(a, n2, stride_along, m, stride_crosswise, first, 1);
        pairwise_blocksum_FLOAT(a + n2 * stride_along, n - n2, stride_along, m, stride_crosswise, second, 1);
        for(index_t j = 0; j < m; j++){          
           output[j*stride_output] = first[j]+second[j];
       }
    }
}


void pairwise_2dsum_FLOAT(const value_t *a, index_t n, index_t stride_along, index_t m, index_t stride_crosswise, value_t *output, index_t stride_output){
    while(m>ROW_COUNT){
        pairwise_blocksum_FLOAT(a, n, stride_along, ROW_COUNT, stride_crosswise, output, stride_output);
        a+=stride_crosswise*ROW_COUNT;
        output+=stride_output*ROW_COUNT; 
        m-=ROW_COUNT; 
    }
    pairwise_blocksum_FLOAT(a, n, stride_along, m, stride_crosswise, output, stride_output);
}


