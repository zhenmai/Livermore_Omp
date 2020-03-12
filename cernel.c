#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

//cIBM  IMPLICIT  REAL*8           (A-H,O-Z)
//C
//C
//C/      PARAMETER( l1=   1001, l2=   101, l1d= 2*1001 )
//C/      PARAMETER( l13= 64, l13h= 64/2, l213= 64+32, l813= 8*64 )
//C/      PARAMETER( l14= 2048, l16= 75, l416= 4*75 , l21= 25)
//C
//INTEGER    E,F,ZONE
//COMMON /ISPACE/ E(96), F(96),
//1  IX(1001), IR(1001), ZONE(300)
//C
//COMMON /SPACE1/ U(1001), V(1001), W(1001),
//1  X(1001), Y(1001), Z(1001), G(1001),
//2  DU1(101), DU2(101), DU3(101), GRD(1001), DEX(1001),
//3  XI(1001), EX(1001), EX1(1001), DEX1(1001),
//4  VX(1001), XX(1001), RX(1001), RH(2048),
//5  VSP(101), VSTP(101), VXNE(101), VXND(101),
//6  VE3(101), VLR(101), VLIN(101), B5(101),
//7  PLAN(300), D(300), SA(101), SB(101)
//C
//COMMON /SPACE2/ P(4,512), PX(25,101), CX(25,101),
//1  VY(101,25), VH(101,7), VF(101,7), VG(101,7), VS(101,7),
//2  ZA(101,7)  , ZP(101,7), ZQ(101,7), ZR(101,7), ZM(101,7),
//3  ZB(101,7)  , ZU(101,7), ZV(101,7), ZZ(101,7),
//4  B(64,64), C(64,64), H(64,64),
//5  U1(5,101,2),  U2(5,101,2),  U3(5,101,2)

//long lw , j , nl1 , nl2 , kx , ky , ip , kn;
//long i1 , j1 , i2 , j2 , nz , ink , jn , kb5i;
//long ii , lb , j4 , ng;
//float tmp , temp;

void InitializeArray(float* arr, size_t size, float val)
{
    // initializing array elements
    for (int i = 0; i < size ; i++){
        *(arr + i) = val;
    }
}

/*
 *******************************************************************
 *   Kernel 1 -- hydro fragment
 *******************************************************************
 *       DO 1 L = 1,Loop
 *       DO 1 k = 1,n
 *  1       X(k)= Q + Y(k)*(R*ZX(k+10) + T*ZX(k+11))
 */
float* Kernel1()
{
    long k, i = 0, n = 1001;
    float q = 0.5, r = 0.2, t = 0.1;
    static float x[1001] = {0}, y[1001] = {0}, z[1001] = {0};
    InitializeArray(y, 1001, 1);
    InitializeArray(z, 1001, 3);
    do {
        #pragma omp parallel for
        for ( k=0 ; k<n ; k++ )
        {
            x[k] = q + y[k]*( r*z[k+10] + t*z[k+11] );
        }
        ++i;
    } while( i < 10000);
    return x;
}

/*
 *******************************************************************
 *   Kernel 2 -- ICCG excerpt (Incomplete Cholesky Conjugate Gradient)
 *******************************************************************
 *    DO 200  L= 1,Loop
 *        II= n
 *     IPNTP= 0
 *222   IPNT= IPNTP
 *     IPNTP= IPNTP+II
 *        II= II/2
 *         i= IPNTP+1
 CDIR$ IVDEP
 *    DO 2 k= IPNT+2,IPNTP,2
 *         i= i+1
 *  2   X(i)= X(k) - V(k)*X(k-1) - V(k+1)*X(k+1)
 *        IF( II.GT.1) GO TO 222
 *200 CONTINUE
 */
float* Kernel2()
{
    long k, ipntp, ipnt, i, ii, n = 97, j = 0;
    static float x[1001] = {0}, v[1001] = {0};
    do {
        ii = n;
        ipntp = 0;
        do {
            ipnt = ipntp;
            ipntp += ii;
            ii /= 2;
            i = ipntp ;
            /* #pragma nohazard */
            for ( k=ipnt+1 ; k<ipntp ; k=k+2 ) {
                i++;
                x[i] = x[k] - v[k  ]*x[k-1] - v[k+1]*x[k+1];
            }
        } while ( ii>0 );
        j++;
    } while( j < 2680 );
    return x;
}


float timedifference_msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000000.0f + (t1.tv_usec - t0.tv_usec);
}

int main(int argc, char *argv[])
{
    if( argc >= 2 )
    {
        omp_set_num_threads( atoi( argv[ 1 ] ) );
    }
    
    float elapsed = 0;
    struct timeval start, stop;
    gettimeofday(&start, 0);
    
    // Calculate the time used for kernel 1
    float* k1 = Kernel1();
    
    gettimeofday(&stop, 0);
    elapsed = timedifference_msec(start, stop);
    printf("Total time taken is %f microseconds \n",elapsed);
    
    float sum = 0;
    for(int i = 0; i < 1001; i++)
    {
        sum += *(k1 + i);
    }
    printf("Total sum for kernel 1: array x: %f \n", sum);
    
    return 0;
}









