#define BLOCKSIZE 128



// needs include of <Python.h> to work
typedef Py_ssize_t index_t;
typedef float value_t;



//naive implementation:
value_t naive_kahan_1dsum_FLOAT(const value_t *a, index_t n, index_t stride){
   index_t i;
   value_t sum = 0.0f;
   value_t c = 0.0f;
   for(i=0;i<n;i++){
        value_t y=a[i*stride]-c;
        value_t t=sum+y;
        c = (t-sum)-y;
        sum = t;
   }
   return sum;
}


value_t kahan_1dsum_FLOAT(const value_t *a, index_t n, index_t stride)
{
    if (n < 8) {
        return naive_kahan_1dsum_FLOAT(a,n,stride);
    }
    
    index_t i;
    value_t sum[9]; 
    value_t c[8] = {0.0f};


    sum[0] = (a[0 * stride]);
    sum[1] = (a[1 * stride]);
    sum[2] = (a[2 * stride]);
    sum[3] = (a[3 * stride]);
    sum[4] = (a[4 * stride]);
    sum[5] = (a[5 * stride]);
    sum[6] = (a[6 * stride]);
    sum[7] = (a[7 * stride]);

    for (i = 8; i < n - (n % 8); i += 8) {
        value_t y[8], t[8];
        y[0] = a[(i + 0) * stride]-c[0];
        y[1] = a[(i + 1) * stride]-c[1];
        y[2] = a[(i + 2) * stride]-c[2];
        y[3] = a[(i + 3) * stride]-c[3];
        y[4] = a[(i + 4) * stride]-c[4];
        y[5] = a[(i + 5) * stride]-c[5];
        y[6] = a[(i + 6) * stride]-c[6];
        y[7] = a[(i + 7) * stride]-c[7];

        t[0] = sum[0]+y[0];
        t[1] = sum[1]+y[1];
        t[2] = sum[2]+y[2];
        t[3] = sum[3]+y[3];
        t[4] = sum[4]+y[4];
        t[5] = sum[5]+y[5];
        t[6] = sum[6]+y[6];
        t[7] = sum[7]+y[7];

        c[0] = (t[0]-sum[0])-y[0];
        c[1] = (t[1]-sum[1])-y[1];
        c[2] = (t[2]-sum[2])-y[2];
        c[3] = (t[3]-sum[3])-y[3];
        c[4] = (t[4]-sum[4])-y[4];
        c[5] = (t[5]-sum[5])-y[5];
        c[6] = (t[6]-sum[6])-y[6];
        c[7] = (t[7]-sum[7])-y[7];

        sum[0] = t[0];
        sum[1] = t[1];
        sum[2] = t[2];
        sum[3] = t[3];
        sum[4] = t[4];
        sum[5] = t[5];
        sum[6] = t[6];
        sum[7] = t[7];
     }
     sum[8] = naive_kahan_1dsum_FLOAT(a+i,n-i,stride);
     return naive_kahan_1dsum_FLOAT(sum,9,1);
}


