#include <math.h>


#define BLOCKSIZE 128



// needs include of <Python.h> to work
typedef Py_ssize_t index_t;
typedef float value_t;



//naive implementation:
value_t naive_neumaier_1dsum_FLOAT(const value_t *a, index_t n, index_t stride){
   index_t i;
   value_t sum = 0.0f;
   value_t c = 0.0f;
   for(i=0;i<n;i++){
        value_t t = sum + a[i*stride];
        if (fabs(sum)>=fabs(a[i*stride])){
            c+=(sum-t)+a[i*stride];
        }
        else{
            c+=(a[i*stride]-t)+sum;
        }
        sum = t;
   }
   return sum+c;
}


value_t neumaier_1dsum_FLOAT(const value_t *a, index_t n, index_t stride)
{
    if (n < 8) {
        return naive_neumaier_1dsum_FLOAT(a,n,stride);
    }
    
    index_t i;
    value_t sum[8]={0.0f}; 
    value_t c[9] = {0.0f};

    //unrolling
    //sum[0] = (a[0 * stride]);
    //sum[1] = (a[1 * stride]);
    //sum[2] = (a[2 * stride]);
    //sum[3] = (a[3 * stride]);
    //sum[4] = (a[4 * stride]);
    //sum[5] = (a[5 * stride]);
    //sum[6] = (a[6 * stride]);
    //sum[7] = (a[7 * stride]);

    for (i = 0; i < n - (n % 8); i += 8) {
        value_t t[8];
        t[0] = sum[0]+a[(i + 0) * stride];
        t[1] = sum[1]+a[(i + 1) * stride];
        t[2] = sum[2]+a[(i + 2) * stride];
        t[3] = sum[3]+a[(i + 3) * stride];
        t[4] = sum[4]+a[(i + 4) * stride];
        t[5] = sum[5]+a[(i + 5) * stride];
        t[6] = sum[6]+a[(i + 6) * stride];
        t[7] = sum[7]+a[(i + 7) * stride];
         
        c[0]+=fabs(sum[0])>=fabs(a[(i+0)*stride]) ? (sum[0]-t[0])+a[(i+0)*stride] : (a[(i+0)*stride]-t[0])+sum[0];
        c[1]+=fabs(sum[1])>=fabs(a[(i+1)*stride]) ? (sum[1]-t[1])+a[(i+1)*stride] : (a[(i+1)*stride]-t[1])+sum[1];
        c[2]+=fabs(sum[2])>=fabs(a[(i+2)*stride]) ? (sum[2]-t[2])+a[(i+2)*stride] : (a[(i+2)*stride]-t[2])+sum[2];
        c[3]+=fabs(sum[3])>=fabs(a[(i+3)*stride]) ? (sum[3]-t[3])+a[(i+3)*stride] : (a[(i+3)*stride]-t[3])+sum[3];
        c[4]+=fabs(sum[4])>=fabs(a[(i+4)*stride]) ? (sum[4]-t[4])+a[(i+4)*stride] : (a[(i+4)*stride]-t[4])+sum[4];
        c[5]+=fabs(sum[5])>=fabs(a[(i+5)*stride]) ? (sum[5]-t[5])+a[(i+5)*stride] : (a[(i+5)*stride]-t[5])+sum[5];
        c[6]+=fabs(sum[6])>=fabs(a[(i+6)*stride]) ? (sum[6]-t[6])+a[(i+6)*stride] : (a[(i+6)*stride]-t[6])+sum[6];
        c[7]+=fabs(sum[7])>=fabs(a[(i+7)*stride]) ? (sum[7]-t[7])+a[(i+7)*stride] : (a[(i+7)*stride]-t[7])+sum[7];

        sum[0] = t[0];
        sum[1] = t[1];
        sum[2] = t[2];
        sum[3] = t[3];
        sum[4] = t[4];
        sum[5] = t[5];
        sum[6] = t[6];
        sum[7] = t[7];
     }
     sum[0] += c[0];
     sum[1] += c[1];
     sum[2] += c[2];
     sum[3] += c[3];
     sum[4] += c[4];
     sum[5] += c[5];
     sum[6] += c[6];
     sum[7] += c[7];

     sum[8] = naive_neumaier_1dsum_FLOAT(a+stride*i,n-i,stride);
     return naive_neumaier_1dsum_FLOAT(sum,9,1);
}


/*

#define ROW_COUNT 32
void neumaier_blocksum_FLOAT(const value_t *a, index_t n, index_t stride_along, index_t m, index_t stride_crosswise, value_t *output, index_t stride_output)
{
    //we trust m to be<=ROW_COUNT
    if (n < 8) {
        index_t j;
        for( j = 0; j < m; j++){
               output[j*stride_output]=naive_neumaier_1dsum_FLOAT(a+j*stride_crosswise,n,stride_along);
        }
    }
    else{
        index_t i,j;
        value_t sum[ROW_COUNT][8]; 
        value_t c[ROW_COUNT][9] = {};

        //unrolling
        for( j = 0; j < m; j++){
            sum[j][0] = (a[0 * stride_along+j*stride_crosswise]);
            sum[j][1] = (a[1 * stride_along+j*stride_crosswise]);
            sum[j][2] = (a[2 * stride_along+j*stride_crosswise]);
            sum[j][3] = (a[3 * stride_along+j*stride_crosswise]);
            sum[j][4] = (a[4 * stride_along+j*stride_crosswise]);
            sum[j][5] = (a[5 * stride_along+j*stride_crosswise]);
            sum[j][6] = (a[6 * stride_along+j*stride_crosswise]);
            sum[j][7] = (a[7 * stride_along+j*stride_crosswise]);
        }

        for (i = 8; i < n - (n % 8); i += 8) {
            value_t t[ROW_COUNT][8];
            for( j = 0; j < m; j++){
                t[j][0] = sum[0]+a[(i + 0) * _along+j*stride_crosswise];
                t[j][1] = sum[1]+a[(i + 1) * _along+j*stride_crosswise];
                t[j][2] = sum[2]+a[(i + 2) * _along+j*stride_crosswise];
                t[j][3] = sum[3]+a[(i + 3) * _along+j*stride_crosswise];
                t[j][4] = sum[4]+a[(i + 4) * _along+j*stride_crosswise];
                t[j][5] = sum[5]+a[(i + 5) * _along+j*stride_crosswise];
                t[j][6] = sum[6]+a[(i + 6) * _along+j*stride_crosswise];
                t[j][7] = sum[7]+a[(i + 7) * _along+j*stride_crosswise];
                 
                c[0]+=fabs(sum[0])>=fabs(a[(i+0)*_along+j*stride_crosswise]) ? (sum[0]-t[j][0])+a[(i+0)*_along+j*stride_crosswise] : (a[(i+0)*_along+j*stride_crosswise]-t[j][0])+sum[0];
                c[1]+=fabs(sum[1])>=fabs(a[(i+1)*_along+j*stride_crosswise]) ? (sum[1]-t[j][1])+a[(i+1)*_along+j*stride_crosswise] : (a[(i+1)*_along+j*stride_crosswise]-t[j][1])+sum[1];
                c[2]+=fabs(sum[2])>=fabs(a[(i+2)*_along+j*stride_crosswise]) ? (sum[2]-t[j][2])+a[(i+2)*_along+j*stride_crosswise] : (a[(i+2)*_along+j*stride_crosswise]-t[j][2])+sum[2];
                c[3]+=fabs(sum[3])>=fabs(a[(i+3)*_along+j*stride_crosswise]) ? (sum[3]-t[j][3])+a[(i+3)*_along+j*stride_crosswise] : (a[(i+3)*_along+j*stride_crosswise]-t[j][3])+sum[3];
                c[4]+=fabs(sum[4])>=fabs(a[(i+4)*_along+j*stride_crosswise]) ? (sum[4]-t[j][4])+a[(i+4)*_along+j*stride_crosswise] : (a[(i+4)*_along+j*stride_crosswise]-t[j][4])+sum[4];
                c[5]+=fabs(sum[5])>=fabs(a[(i+5)*_along+j*stride_crosswise]) ? (sum[5]-t[j][5])+a[(i+5)*_along+j*stride_crosswise] : (a[(i+5)*_along+j*stride_crosswise]-t[j][5])+sum[5];
                c[6]+=fabs(sum[6])>=fabs(a[(i+6)*_along+j*stride_crosswise]) ? (sum[6]-t[j][6])+a[(i+6)*_along+j*stride_crosswise] : (a[(i+6)*_along+j*stride_crosswise]-t[j][6])+sum[6];
                c[7]+=fabs(sum[7])>=fabs(a[(i+7)*_along+j*stride_crosswise]) ? (sum[7]-t[j][7])+a[(i+7)*_along+j*stride_crosswise] : (a[(i+7)*_along+j*stride_crosswise]-t[j][7])+sum[7];

                sum[j][0] = t[j][0];
                sum[j][1] = t[j][1];
                sum[j][2] = t[j][2];
                sum[j][3] = t[j][3];
                sum[j][4] = t[j][4];
                sum[j][5] = t[j][5];
                sum[j][6] = t[j][6];
                sum[j][7] = t[j][7];
            }
         }
         
         //sums are in c:
         for( j = 0; j < m; j++){
            c[j][8] = naive_neumaier_1dsum_FLOAT(a+i*stride_along+j*stride_crosswise,n-i,stride_along);
            output[j*stride_output] = naive_neumaier_1dsum_FLOAT(c[j],9,1);
         }
    }
}

void neumaier_2dsum_FLOAT(const value_t *a, index_t n, index_t stride_along, index_t m, index_t stride_crosswise, value_t *output, index_t stride_output){
    while(m>ROW_COUNT){
        neumaier_blocksum_FLOAT(a, n, stride_along, ROW_COUNT, stride_crosswise, output, stride_output);
        a+=stride_crosswise*ROW_COUNT;
        output+=stride_output*ROW_COUNT; 
        m-=ROW_COUNT; 
    }
    neumaier_blocksum_FLOAT(a, n, stride_along, m, stride_crosswise, output, stride_output);
}


*/
