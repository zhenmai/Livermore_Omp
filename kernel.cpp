#include <iostream>
#include <cstring>
#include <string>
#include <cmath>
#include <vector>
#include <chrono>
#include <ctime>
#include <numeric>  // std::accumulate()
#include <omp.h>

// const iteration time for every loop
extern const int iter = 100;

template <typename T> 
double CalculateSum(const std::vector<T> input)
{
    double sum = 0;
    // sum up all the elements in the vector
    #pragma omp parallel for reduction(+:sum)
    for (unsigned int i = 0; i < input.size(); i++)
    {
        sum += input[i];
    }
    return sum;
}

template <typename T> 
void InitializeLoop(std::vector<T>& input, const float mul)
{
    // Initialize the loop with different values
    #pragma omp parallel for
    for (unsigned i = 0; i < input.size(); i++)
    {
        input[i] = i * mul;
    }
}

template <typename T> 
void Initialize2DLoop(std::vector<std::vector<T> >& input, const float mul)
{
    // Initialize the loop with different values
    #pragma omp parallel for
    for (unsigned int i = 0; i < input.size(); i++)
    {
        for (unsigned int j = 0; j < input[i].size(); j++)
        {
            input[i][j] = i * mul * j;
        }
    }
}

template <typename T> 
void Print1DArray(const std::vector<T> input)
{
    for (unsigned int i = 0; i < input.size(); i++)
    {
        std::cout << input[i] << std::endl;
    }
}

template <typename T> 
void Print2DArray(const std::vector<std::vector<T> > input)
{
    for (unsigned int i = 0; i < input.size(); i++)
    {
        for (unsigned int j = 0; j < input[i].size(); j++)
        {
            std::cout << input[i][j] << std::endl;
        }
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
std::vector<float> Kernel1()
{
    int i = 0, n = 1000000;
    float q = 0.05, r = 0.02, t = 0.01;
    std::vector<float> x(n, 0);
    std::vector<float> y(n, 0);
    std::vector<float> z(n + 11, 0);
    InitializeLoop(x, 0.001);
    InitializeLoop(y, 0.0003);
    InitializeLoop(z, 0.0005);

    do {
        #pragma omp parallel for
        for (int k = 0; k < n; k++)
        {
            x[k] = q + y[k] * (r * z[k+10] + t * z[k+11]);
        }
    } while( ++i < iter);
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
std::vector<float> Kernel2()
{
    int ipntp, ipnt, i, ii, n = 10000, j = 0;
    std::vector<float> x(2*n, 0);
    std::vector<float> v(2*n, 0);
    InitializeLoop(x, -0.002);
    InitializeLoop(v, 0.0007);
    do 
    {
        ii = n;
        ipntp = 0;
        do 
        {
            ipnt = ipntp;
            ipntp += ii;
            ii /= 2;
            i = ipntp ;
            #pragma omp parallel for
            // This loop is not suitable for parallelization because the x[i] and x[k] are dependent 
            for (int k = ipnt+1; k< ipntp ; k = k+2) 
            {
                i++;
                x[i] = x[k] - v[k] * x[k-1] - v[k+1] * x[k+1];
            }
        } while (ii > 0);
    } while( ++j < 1 );
    return x;
}
/*
*******************************************************************
*   Kernel 3 -- inner product
*******************************************************************
*    DO 3 L= 1,Loop
*         Q= 0.0
*    DO 3 k= 1,n
*  3      Q= Q + Z(k)*X(k)
*/
double Kernel3()
{
    double q = 0.0;
    int i = 0, n = 1000000;
    std::vector<float> x(n, 0);
    std::vector<float> z(n, 0);
    InitializeLoop(x, 0.0001);
    InitializeLoop(z, 0.0015);

    do 
    {
       #pragma omp parallel for reduction(+:q)
       for (int k = 0; k < n; k++) 
       {
           q += z[k] * x[k];    
       }
    } while( ++i < iter );
    return q;
}

/*
*******************************************************************
*   Kernel 4 -- banded linear equations
*******************************************************************
*            m= (1001-7)/2
*    DO 444  L= 1,Loop
*    DO 444  k= 7,1001,m
*           lw= k-6
*         temp= X(k-1)
CDIR$ IVDEP
*    DO   4  j= 5,n,5
*       temp  = temp   - XZ(lw)*Y(j)
*  4        lw= lw+1
*       X(k-1)= Y(5)*temp
*444 CONTINUE
*/
std::vector<float> Kernel4()
{
    int j = 0, n = 1000000;
    int m = (1001 - 7)/2;
    std::vector<float> x(n, 0);
    std::vector<float> y(n, 0);
    InitializeLoop(x, 0.01);
    InitializeLoop(y, 0.0035);

    do {
        for (int k = 6; k < 10000; k = k + m) 
        {
            int lw = k - 6;
            float temp = x[k-1];
            #pragma omp parallel for reduction(-:temp)
            for (int j = 4; j < n; j = j + 5) 
            {
                temp -= x[lw] * y[j];
                lw++;
            }
            x[k-1] = y[4] * temp;
        }
    } while( ++j < 1 );
    return x;
}

/*
*******************************************************************
*   Kernel 5 -- tri-diagonal elimination, below diagonal
*******************************************************************
*    DO 5 L = 1,Loop
*    DO 5 i = 2,n
*  5    X(i)= Z(i)*(Y(i) - X(i-1))
*/

std::vector<float> Kernel5()
{
    int j = 0, n = 10000;
    std::vector<float> x(n, 0);
    std::vector<float> y(n, 0);
    std::vector<float> z(n, 0);
    InitializeLoop(y, 0.0000305);
    InitializeLoop(z, 0.0000023);
    do {
        // This loop can't be parallelize because the loop dependence on x[i] and x[i-1]
        #pragma omp parallel for
        for (int i = 1; i < n; i++ ) 
        {
           x[i] = z[i] * (y[i] - x[i-1]);
        }
    } while(++j < 1);
    // Uncomment the function Print1DArrray could print out each array element
    // Compared to results with different threads we will find the differences
    // Print1DArray(x); 
    return x;
}


/*
*******************************************************************
*   Kernel 6 -- general linear recurrence equations
*******************************************************************
*    DO  6  L= 1,Loop
*    DO  6  i= 2,n
*    DO  6  k= 1,i-1
*        W(i)= W(i)  + B(i,k) * W(i-k)
*  6 CONTINUE
*/

std::vector<float> Kernel6()
{
    int j = 0, n = 100;
    std::vector<float> w(n, 0);
    InitializeLoop(w, 0.00000012);
    std::vector<std::vector<float> > b(n, std::vector<float>(n, 0));
    Initialize2DLoop(b, 0.000007);
    do {
        // This loop is not suitable for parallelization because the loop dependence on w[i] and w[(i-k)-1]
        #pragma omp parallel for
        for (int i = 1; i < n; i++) 
        {
            w[i] = 0.0100;
            for (int k = 0; k < i; k++) 
            {
                w[i] += b[k][i] * w[(i-k)-1];
            }
        } 
    } while(++j < 1);
    // Uncomment the function Print1DArrray could print out each array element
    // Compared to results with different threads we will find the differences
    // Print1DArray(w);
    return w;
}

/*
*******************************************************************
*   Kernel 7 -- equation of state fragment
*******************************************************************
*    DO 7 L= 1,Loop
*    DO 7 k= 1,n
*      X(k)=     U(k  ) + R*( Z(k  ) + R*Y(k  )) +
*   .        T*( U(k+3) + R*( U(k+2) + R*U(k+1)) +
*   .        T*( U(k+6) + Q*( U(k+5) + Q*U(k+4))))
*  7 CONTINUE
*/
std::vector<float> Kernel7()
{
    int i = 0, n = 1000000;
    float q = 0.5, r = 0.2, t = 0.1;
    std::vector<float> x(n, 0);
    std::vector<float> y(n, 0);
    std::vector<float> z(n, 0);
    std::vector<float> u(n + 6, 0);
    InitializeLoop(x, 0.0001);
    InitializeLoop(y, 0.00023);
    InitializeLoop(z, 0.0016);
    InitializeLoop(u, 0.0021);

    do {
        #pragma omp parallel for
        for (int k = 0; k < n; k++) 
        {
           x[k] = u[k] + r * (z[k] + r * y[k]) +
                t * (u[k+3] + r * (u[k+2] + r * u[k+1]) +
                    t * (u[k+6] + q * (u[k+5] + q * u[k+4])));
        }
    } while( ++i < iter );
    return x;
}

/*
*******************************************************************
*   Kernel 8 -- ADI integration
*******************************************************************
*    DO  8      L = 1,Loop
*             nl1 = 1
*             nl2 = 2
*    DO  8     kx = 2,3
CDIR$ IVDEP
*    DO  8     ky = 2,n
*          DU1(ky)=U1(kx,ky+1,nl1)  -  U1(kx,ky-1,nl1)
*          DU2(ky)=U2(kx,ky+1,nl1)  -  U2(kx,ky-1,nl1)
*          DU3(ky)=U3(kx,ky+1,nl1)  -  U3(kx,ky-1,nl1)
*    U1(kx,ky,nl2)=U1(kx,ky,nl1) +A11*DU1(ky) +A12*DU2(ky) +A13*DU3(ky)
*   .       + SIG*(U1(kx+1,ky,nl1) -2.*U1(kx,ky,nl1) +U1(kx-1,ky,nl1))
*    U2(kx,ky,nl2)=U2(kx,ky,nl1) +A21*DU1(ky) +A22*DU2(ky) +A23*DU3(ky)
*   .       + SIG*(U2(kx+1,ky,nl1) -2.*U2(kx,ky,nl1) +U2(kx-1,ky,nl1))
*    U3(kx,ky,nl2)=U3(kx,ky,nl1) +A31*DU1(ky) +A32*DU2(ky) +A33*DU3(ky)
*   .       + SIG*(U3(kx+1,ky,nl1) -2.*U3(kx,ky,nl1) +U3(kx-1,ky,nl1))
*  8 CONTINUE
*/
void Kernel8()
{
   //  int i = 0, n = 1000;
   //  float du1[n] = {0.1};
   //  int u1[2][1000][4] = {3, 4, 2, 3, 0, -3, 9, 11, 23, 12, 23, 
   //               2, 13, 4, 56, 3, 5, 9, 3, 5, 5, 1, 4, 9};

   //  do {
   //  int nl1 = 0;
   //  int nl2 = 1;
   //  for (int kx = 1; kx < 3; kx++)
   //  {
   //     for (int ky = 1; ky < n; ky++) 
   //     {
   //          du1[ky] = u1[nl1][ky+1][kx] - u1[nl1][ky-1][kx];
   //          du2[ky] = u2[nl1][ky+1][kx] - u2[nl1][ky-1][kx];
   //          du3[ky] = u3[nl1][ky+1][kx] - u3[nl1][ky-1][kx];
   //          u1[nl2][ky][kx] = u1[nl1][ky][kx] + a11 * du1[ky] + a12 * du2[ky] + a13 * du3[ky] + sig *
   //              (u1[nl1][ky][kx+1] - 2.0 * u1[nl1][ky][kx] + u1[nl1][ky][kx-1]);
   //          u2[nl2][ky][kx] = u2[nl1][ky][kx] + a21 * du1[ky] + a22 * du2[ky] + a23 * du3[ky] + sig *
   //              (u2[nl1][ky][kx+1] - 2.0 * u2[nl1][ky][kx] + u2[nl1][ky][kx-1]);
   //          u3[nl2][ky][kx] = u3[nl1][ky][kx] + a31 * du1[ky] + a32 * du2[ky] + a33 * du3[ky] + sig *
   //              (u3[nl1][ky][kx+1] - 2.0 * u3[nl1][ky][kx] + u3[nl1][ky][kx-1]);
   //     }
   //  }
   // } while(i++ < iter);
}

/*
*******************************************************************
*   Kernel 9 -- integrate predictors
*******************************************************************
*    DO 9  L = 1,Loop
*    DO 9  i = 1,n
*    PX( 1,i)= DM28*PX(13,i) + DM27*PX(12,i) + DM26*PX(11,i) +
*   .          DM25*PX(10,i) + DM24*PX( 9,i) + DM23*PX( 8,i) +
*   .          DM22*PX( 7,i) +  C0*(PX( 5,i) +      PX( 6,i))+ PX( 3,i)
*  9 CONTINUE
*/
void Kernel9()
{
    int j = 0, n = 100000;
    float dm28 = 0.05, dm27 = 0.02, dm26 = 0.012, dm25 = 0.037, 
        dm24 = 0.04, dm23 = 0.09, dm22 = 0.024, c0 = 0.224;
    std::vector<std::vector<float> > px(n, std::vector<float>(12, 0));
    Initialize2DLoop(px, 0.00002);

    do {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) 
        {
            px[i][0] = dm28 * px[i][12] + dm27 * px[i][11] + dm26 * px[i][10] +
                  dm25 * px[i][9] + dm24 * px[i][8] + dm23 * px[i][7] +
                  dm22 * px[i][6] + c0 * (px[i][4] + px[i][5]) + px[i][2];
        }
    } while( ++j < iter);
}

/*
*******************************************************************
*   Kernel 10 -- difference predictors
*******************************************************************
*    DO 10  L= 1,Loop
*    DO 10  i= 1,n
*    AR      =      CX(5,i)
*    BR      = AR - PX(5,i)
*    PX(5,i) = AR
*    CR      = BR - PX(6,i)
*    PX(6,i) = BR
*    AR      = CR - PX(7,i)
*    PX(7,i) = CR
*    BR      = AR - PX(8,i)
*    PX(8,i) = AR
*    CR      = BR - PX(9,i)
*    PX(9,i) = BR
*    AR      = CR - PX(10,i)
*    PX(10,i)= CR
*    BR      = AR - PX(11,i)
*    PX(11,i)= AR
*    CR      = BR - PX(12,i)
*    PX(12,i)= BR
*    PX(14,i)= CR - PX(13,i)
*    PX(13,i)= CR
* 10 CONTINUE
*/
void Kernel10()
{
    int j = 0, n = 100000;
    std::vector<std::vector<float> > cx(n, std::vector<float>(4, 1000));
    std::vector<std::vector<float> > px(n, std::vector<float>(13, 0));
    Initialize2DLoop(px, 0.02);

    do {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) 
        {
            float cx4 = cx[i][4];
            float px4 = px[i][4];
            float px5 = px[i][5];
            float px6 = px[i][6];
            float px7 = px[i][7];
            float px8 = px[i][8];
            float px9 = px[i][9];
            float px10 = px[i][10];
            float px11 = px[i][11];
            float px12 = px[i][12];
         
            px[i][4] = cx4;
            px[i][5] = cx4 - px4;
            px[i][6] = cx4 - px4 - px5;
            px[i][7] = cx4 - px4 - px5 - px6;
            px[i][8] = cx4 - px4 - px5 - px6 - px7;
            px[i][9] = cx4 - px4 - px5 - px6 - px7 - px8;
            px[i][10] = cx4 - px4 - px5 - px6 - px7 - px8 - px9;
            px[i][11] = cx4 - px4 - px5 - px6 - px7 - px8 - px9 - px10;
            px[i][13] = cx4 - px4 - px5 - px6 - px7 - px8 - px9 - px10 - px11 - px12;
            px[i][12] = cx4 - px4 - px5 - px6 - px7 - px8 - px9 - px10 - px11;
        }
    } while( ++j < iter );
    // Print2DArray(px);
}

/*
*******************************************************************
*   Kernel 11 -- first sum
*******************************************************************
*    DO 11 L = 1,Loop
*        X(1)= Y(1)
*    DO 11 k = 2,n
* 11     X(k)= X(k-1) + Y(k)
*/
std::vector<float> Kernel11()
{
    int i = 0, n = 1000;
    std::vector<float> x(n, 0);
    std::vector<float> y(n, 0);
    InitializeLoop(x, 0.0001);
    InitializeLoop(y, 0.00023);
    do 
    {
        x[0] = y[0];
        // This loop can't be parallelize because the loop dependence on x[k] and x[k-1]
        #pragma omp parallel for
        for (int k = 1; k < n; k++) 
        {
           x[k] = x[k-1] + y[k];
        }
    } while(++i < 1);
    // Uncomment the function Print1DArrray could print out each array element
    // Compared to results with different threads we will find the differences
    // Print1DArray(x); 
    return x;
}

/*
*******************************************************************
*   Kernel 12 -- first difference
*******************************************************************
*    DO 12 L = 1,Loop
*    DO 12 k = 1,n
* 12     X(k)= Y(k+1) - Y(k)
*/
std::vector<float> Kernel12()
{
    int i = 0, n = 1000000;
    std::vector<float> x(n, 0);
    std::vector<float> y(n+1, 0);
    InitializeLoop(x, 0.0001);
    InitializeLoop(y, 0.00023);
    do 
    {
        #pragma omp parallel for
        for (int k = 1; k < n; k++) 
        {
           x[k] = y[k+1] - y[k];
        }
    } while(++i < iter);
    // Print1DArray(x); 
    return x;
}

/*
*******************************************************************
*   Kernel 13 -- 2-D PIC (Particle In Cell)
*******************************************************************
*    DO  13     L= 1,Loop
*    DO  13    ip= 1,n
*              i1= P(1,ip)
*              j1= P(2,ip)
*              i1=        1 + MOD2N(i1,64)
*              j1=        1 + MOD2N(j1,64)
*         P(3,ip)= P(3,ip)  + B(i1,j1)
*         P(4,ip)= P(4,ip)  + C(i1,j1)
*         P(1,ip)= P(1,ip)  + P(3,ip)
*         P(2,ip)= P(2,ip)  + P(4,ip)
*              i2= P(1,ip)
*              j2= P(2,ip)
*              i2=            MOD2N(i2,64)
*              j2=            MOD2N(j2,64)
*         P(1,ip)= P(1,ip)  + Y(i2+32)
*         P(2,ip)= P(2,ip)  + Z(j2+32)
*              i2= i2       + E(i2+32)
*              j2= j2       + F(j2+32)
*        H(i2,j2)= H(i2,j2) + 1.0
* 13 CONTINUE
*/

void Kernel13()
{
#if 1
    int i1, j1, i2, j2;
    int i = 0, n = 10000;
    std::vector<int> y(n+32, 1);
    std::vector<int> z(n+32, 1);
    std::vector<int> e(n+32, 1);
    std::vector<int> f(n+32, 1);
    InitializeLoop(y, 1);
    InitializeLoop(z, 1);
    InitializeLoop(e, 1);
    InitializeLoop(f, 1);
    std::vector<std::vector<int> > b(n, std::vector<int>(n, 1));
    std::vector<std::vector<int> > c(n, std::vector<int>(n, 3));
    std::vector<std::vector<int> > p(n, std::vector<int>(4, 0));
    std::vector<std::vector<int> > h(n, std::vector<int>(n, 0));
    Initialize2DLoop(p, 0.8);
    Initialize2DLoop(h, 0.2);

    do {
        #pragma omp parallel for private(i1,j1,j2,i2)
        for (int ip = 0 ; ip < n; ip++) 
        {
           i1 = p[ip][0];
           j1 = p[ip][1];
           i1 &= 64-1;
           j1 &= 64-1;
           p[ip][2] += b[j1][i1];
           p[ip][3] += c[j1][i1];
           p[ip][0] += p[ip][2];
           p[ip][1] += p[ip][3];
           i2 = p[ip][0];
           j2 = p[ip][1];
           i2 = ( i2 & 64-1 ) - 1 ;
           j2 = ( j2 & 64-1 ) - 1 ;
           p[ip][0] += y[i2+32];
           p[ip][1] += z[j2+32];
           i2 += e[i2+32];
           j2 += f[j2+32];
           #pragma atomic
           h[j2][i2] += 1.0;
        }
    } while( i++ < iter );
    // Uncomment the function Print1DArrray could print out each array element
    // Compared to results with different threads we will find the differences
    // Print2DArray(h); 
#endif
}


/*
 *******************************************************************
 *   Kernel 14 -- 1-D PIC (Particle In Cell)
 *******************************************************************
 *    DO   14   L= 1,Loop
 *    DO   141  k= 1,n
 *          VX(k)= 0.0
 *          XX(k)= 0.0
 *          IX(k)= INT(  GRD(k))
 *          XI(k)= REAL( IX(k))
 *         EX1(k)= EX   ( IX(k))
 *        DEX1(k)= DEX  ( IX(k))
 *41  CONTINUE
 *    DO   142  k= 1,n
 *          VX(k)= VX(k) + EX1(k) + (XX(k) - XI(k))*DEX1(k)
 *          XX(k)= XX(k) + VX(k)  + FLX
 *          IR(k)= XX(k)
 *          RX(k)= XX(k) - IR(k)
 *          IR(k)= MOD2N(  IR(k),2048) + 1
 *          XX(k)= RX(k) + IR(k)
 *42  CONTINUE
 *    DO  14    k= 1,n
 *    RH(IR(k)  )= RH(IR(k)  ) + 1.0 - RX(k)
 *    RH(IR(k)+1)= RH(IR(k)+1) + RX(k)
 *14  CONTINUE
 */
void Kernel14()
{
#if 1
   int i = 0, n = 100000, flx = 1;
   std::vector<int> ir(n, 1.0);
   std::vector<long> ix(n, 1.0);
   std::vector<float> vx(n, 1.0);
   std::vector<float> xx(n, 1.0);
   std::vector<float> xi(n, 1.0);
   std::vector<float> grd(n, 1.0);
   std::vector<float> ex(n, 1.0);
   std::vector<float> dex(n, 1.0);
   std::vector<float> ex1(n, 1.0);
   std::vector<float> dex1(n, 1.0);
   std::vector<float> rx(n, 1.0);
   std::vector<float> rh(n, 1.0); 

   do {
      #pragma omp parallel for
      for ( int k=0 ; k<n ; k++ ) {
           vx[k] = 0.0;
           xx[k] = 0.0;
           ix[k] = (long) grd[k];
           xi[k] = (float) ix[k];
           ex1[k] = ex[ ix[k] - 1 ];
           dex1[k] = dex[ ix[k] - 1 ];
       }
      #pragma omp parallel for
       for ( int k=0 ; k<n ; k++ ) {
           vx[k] = vx[k] + ex1[k] + ( xx[k] - xi[k] )*dex1[k];
           xx[k] = xx[k] + vx[k]  + flx;
           ir[k] = xx[k];
           rx[k] = xx[k] - ir[k];
           ir[k] = ( ir[k] & 2048-1 ) + 1;
           xx[k] = rx[k] + ir[k];
       }
      #pragma omp parallel for
       for ( int k=0 ; k<n ; k++ ) {
            rh[ ir[k]-1 ] += 1.0 - rx[k];
            rh[ ir[k]   ] += rx[k];
       } 
   } while( i++ < iter);
#endif
}


/*
 *******************************************************************
 *   Kernel 15 -- Casual Fortran.  Development version
 *******************************************************************
 *      DO 45  L = 1,Loop
 *             NG= 7
 *             NZ= n
 *             AR= 0.053
 *             BR= 0.073
 * 15   DO 45  j = 2,NG
 *      DO 45  k = 2,NZ
 *             IF( j-NG) 31,30,30
 * 30     VY(k,j)= 0.0
 *                 GO TO 45
 * 31          IF( VH(k,j+1) -VH(k,j)) 33,33,32
 * 32           T= AR
 *                 GO TO 34
 * 33           T= BR
 * 34          IF( VF(k,j) -VF(k-1,j)) 35,36,36
 * 35           R= MAX( VH(k-1,j), VH(k-1,j+1))
 *              S= VF(k-1,j)
 *                 GO TO 37
 * 36           R= MAX( VH(k,j),   VH(k,j+1))
 *              S= VF(k,j)
 * 37     VY(k,j)= SQRT( VG(k,j)**2 +R*R)*T/S
 * 38          IF( k-NZ) 40,39,39
 * 39     VS(k,j)= 0.
 *                 GO TO 45
 * 40          IF( VF(k,j) -VF(k,j-1)) 41,42,42
 * 41           R= MAX( VG(k,j-1), VG(k+1,j-1))
 *              S= VF(k,j-1)
 *              T= BR
 *                 GO TO 43
 * 42           R= MAX( VG(k,j),   VG(k+1,j))
 *              S= VF(k,j)
 *              T= AR
 * 43     VS(k,j)= SQRT( VH(k,j)**2 +R*R)*T/S
 * 45    CONTINUE
 */
std::vector<std::vector<float>> Kernel15()
{
   int i = 0, n = 100000;
   std::vector<std::vector<float>> vy(8, std::vector<float> (n+1, 1.0));
   std::vector<std::vector<float>> vs(8, std::vector<float> (n+1, 1.0));
   do{
      int ng = 7;
      int nz = n;
      float ar = 0.053;
      float br = 0.073;
      std::vector<std::vector<float>> vh(8, std::vector<float> (n+1, 1.5));
      std::vector<std::vector<float>> vf(8, std::vector<float> (n+1, 1.25));
      std::vector<std::vector<float>> vg(8, std::vector<float> (n+1, 2.25));
      #pragma omp parallel for collapse(2) 
      for ( int j=1 ; j<ng ; j++ ) {
           for ( int k=1 ; k<nz ; k++ ) {
               float t,r,s;
               if ( (j+1) >= ng ) {
                   vy[j][k] = 0.0;
                   continue;
               }
               if ( vh[j+1][k] > vh[j][k] ) {
                   t = ar;
               }
               else {
                   t = br;
               }
               if ( vf[j][k] < vf[j][k-1] ) {
                   if ( vh[j][k-1] > vh[j+1][k-1] )
                       r = vh[j][k-1];
                   else
                       r = vh[j+1][k-1];
                   s = vf[j][k-1];
               }
               else {
                   if ( vh[j][k] > vh[j+1][k] )
                       r = vh[j][k];
                   else
                       r = vh[j+1][k];
                   s = vf[j][k];
               }
               vy[j][k] = sqrt( vg[j][k]*vg[j][k] + r*r )* t/s;
               if ( (k+1) >= nz ) {
                
                  
                  vs[j][k] = 0.0;
                   continue;
               }
               if ( vf[j][k] < vf[j-1][k] ) {
                   if ( vg[j-1][k] > vg[j-1][k+1] )
                       r = vg[j-1][k];
                   else
                       r = vg[j-1][k+1];
                   s = vf[j-1][k];
                   t = br;
               }
               else {
                   if ( vg[j][k] > vg[j][k+1] )
                       r = vg[j][k];
                   else
                       r = vg[j][k+1];
                   s = vf[j][k];
                   t = ar;
               }
               vs[j][k] = sqrt( vh[j][k]*vh[j][k] + r*r )* t / s;
           }
       }

   }while(i++ < iter);
   return vs;
}


/*
 *******************************************************************
 *   Kernel 16 -- Monte Carlo search loop
 *******************************************************************
 *          II= n/3
 *          LB= II+II
 *          k2= 0
 *          k3= 0
 *    DO 485 L= 1,Loop
 *           m= 1
 *405       i1= m
 *410       j2= (n+n)*(m-1)+1
 *    DO 470 k= 1,n
 *          k2= k2+1
 *          j4= j2+k+k
 *          j5= ZONE(j4)
 *          IF( j5-n      ) 420,475,450
 *415       IF( j5-n+II   ) 430,425,425
 *420       IF( j5-n+LB   ) 435,415,415
 *425       IF( PLAN(j5)-R) 445,480,440
 *430       IF( PLAN(j5)-S) 445,480,440
 *435       IF( PLAN(j5)-T) 445,480,440
 *440       IF( ZONE(j4-1)) 455,485,470
 *445       IF( ZONE(j4-1)) 470,485,455
 *450       k3= k3+1
 *          IF( D(j5)-(D(j5-1)*(T-D(j5-2))**2+(S-D(j5-3))**2
 *   .                        +(R-D(j5-4))**2)) 445,480,440
 *455        m= m+1
 *          IF( m-ZONE(1) ) 465,465,460
 *460        m= 1
 *465       IF( i1-m) 410,480,410
 *470 CONTINUE
 *475 CONTINUE
 *480 CONTINUE
 *485 CONTINUE
 */
int Kernel16()
{
   int i = 0, n = 10000;
   int ii = n / 3;
   int lb = ii + ii;
   int k3 = 0, k2 = 0;

// change the 0 to 1 could see the code but it is not suitable for parallelism
#if 0
   do {
      int i1 = 1, m = 1;
      bool finish = false;
      while( !finish ){
         int j2 = ( n + n )*( m - 1 ) + 1;
         #pragma omp parallel for ordered schedule(dynamic)
         for ( int k=1 ; k<=n ; k++ ) {
              #pragma omp cancellation point for
              k2++;
              int j4 = j2 + k + k;
              int j5 = zone[j4-1];
              float tmp;
              if ( j5 < n ) {
                  if ( j5+lb < n ) {              /* 420 */
                      tmp = plan[j5-1] - t;       /* 435 */
                  } else {
                      if ( j5+ii < n ) {          /* 415 */
                          tmp = plan[j5-1] - s;   /* 430 */
                      } else {
                          tmp = plan[j5-1] - r;   /* 425 */
                      }
                  }
              } else if( j5 == n ) {
                  finish = true;                          /* 475 */
                  #pragma omp cancel for
                  continue;
              } else {
                  k3++;                           /* 450 */
                  tmp=(d[j5-1]-(d[j5-2]*(t-d[j5-3])*(t-d[j5-3])+(s-d[j5-4])*
                                (s-d[j5-4])+(r-d[j5-5])*(r-d[j5-5])));
              }
              if ( tmp < 0.0 ) {
                  if ( zone[j4-2] < 0 )           /* 445 */
                      continue;                   /* 470 */
                  else if ( !zone[j4-2] ) {
                      #pragma omp cancel for
                      finish = true;                      /* 480 */
                      continue;
                  }
              } else if ( tmp ) {
                  if ( zone[j4-2] > 0 )           /* 440 */
                      continue;                   /* 470 */
                  else if ( !zone[j4-2] ) {
                      #pragma omp cancel for
                      finish = true;                      /* 480 */
                     continue;
                  }
              } else {
                 #pragma omp cancel for
                 finish = true;                       /* 485 */
                 continue;
              }

              #pragma omp single
              {
                 m++;                                /* 455 */
                 #pragma omp cancel for
                 if ( m > zone[0] ) {
                    finish = true;
                 }
              }
         }
      }
   } while( i++ < iter );
#endif
   return 0;
}


/*
 *******************************************************************
 *   Kernel 17 -- implicit, conditional computation
 *******************************************************************
 *          DO 62 L= 1,Loop
 *                i= n
 *                j= 1
 *              INK= -1
 *            SCALE= 5./3.
 *              XNM= 1./3.
 *               E6= 1.03/3.07
 *                   GO TO 61
 *60             E6= XNM*VSP(i)+VSTP(i)
 *          VXNE(i)= E6
 *              XNM= E6
 *           VE3(i)= E6
 *                i= i+INK
 *               IF( i.EQ.j) GO TO  62
 *61             E3= XNM*VLR(i) +VLIN(i)
 *             XNEI= VXNE(i)
 *          VXND(i)= E6
 *              XNC= SCALE*E3
 *               IF( XNM .GT.XNC) GO TO  60
 *               IF( XNEI.GT.XNC) GO TO  60
 *           VE3(i)= E3
 *               E6= E3+E3-XNM
 *          VXNE(i)= E3+E3-XNEI
 *              XNM= E6
 *                i= i+INK
 *               IF( i.NE.j) GO TO 61
 * 62 CONTINUE
 */
std::vector<float> Kernel17()
{
    // This kernel loop is not suitable for parallel execution
    int ii = 0, n = 100000;
    std::vector<float> vsp(n, 1.2);
    std::vector<float> vstp(n, 0.3);
    std::vector<float> vxne(n, 0.5);
    std::vector<float> ve3(n, 2.3);
    std::vector<float> vlr(n, 0.2);
    std::vector<float> vlin(n, 1.3);
    std::vector<float> vxnd(n, 3.0);
    do{
      float scale = 5.0 / 3.0;
      float xnm = 1.0 / 3.0;
      float e6 = 1.03 / 3.07;
      //The following parallel optimization doesn't work since the current 
      //iteration relies on xnm and e6, which are calculated from previous
      //iteration.
      #pragma omp parallel for
      for(int i = n-1; i > 0; i--) {
         float e3 = xnm*vlr[i] + vlin[i];
         float xnei = vxne[i];
         vxnd[i] = e6;
         float xnc = scale*e3;
         if( xnm > xnc || xnei > xnc ) {
            e6 = xnm*vsp[i] + vstp[i]; 
            vxne[i] = e6;
            xnm = e6;
            ve3[i] = e6;
         } else {
            ve3[i] = e3;
            e6 = e3 + e3 - xnm;
            vxne[i] = e3 + e3 - xnei;
            xnm = e6; 
         }
      }
   } while ( ii++ < iter);
   return ve3;
}


/*
 *******************************************************************
 *   Kernel 18 - 2-D explicit hydrodynamics fragment
 *******************************************************************
 *       DO 75  L= 1,Loop
 *              T= 0.0037
 *              S= 0.0041
 *             KN= 6
 *             JN= n
 *       DO 70  k= 2,KN
 *       DO 70  j= 2,JN
 *        ZA(j,k)= (ZP(j-1,k+1)+ZQ(j-1,k+1)-ZP(j-1,k)-ZQ(j-1,k))
 *   .            *(ZR(j,k)+ZR(j-1,k))/(ZM(j-1,k)+ZM(j-1,k+1))
 *        ZB(j,k)= (ZP(j-1,k)+ZQ(j-1,k)-ZP(j,k)-ZQ(j,k))
 *   .            *(ZR(j,k)+ZR(j,k-1))/(ZM(j,k)+ZM(j-1,k))
 * 70    CONTINUE
 *       DO 72  k= 2,KN
 *       DO 72  j= 2,JN
 *        ZU(j,k)= ZU(j,k)+S*(ZA(j,k)*(ZZ(j,k)-ZZ(j+1,k))
 *   .                    -ZA(j-1,k) *(ZZ(j,k)-ZZ(j-1,k))
 *   .                    -ZB(j,k)   *(ZZ(j,k)-ZZ(j,k-1))
 *   .                    +ZB(j,k+1) *(ZZ(j,k)-ZZ(j,k+1)))
 *        ZV(j,k)= ZV(j,k)+S*(ZA(j,k)*(ZR(j,k)-ZR(j+1,k))
 *   .                    -ZA(j-1,k) *(ZR(j,k)-ZR(j-1,k))
 *   .                    -ZB(j,k)   *(ZR(j,k)-ZR(j,k-1))
 *   .                    +ZB(j,k+1) *(ZR(j,k)-ZR(j,k+1)))
 * 72    CONTINUE
 *       DO 75  k= 2,KN
 *       DO 75  j= 2,JN
 *        ZR(j,k)= ZR(j,k)+T*ZU(j,k)
 *        ZZ(j,k)= ZZ(j,k)+T*ZV(j,k)
 * 75    CONTINUE
 */

std::vector<std::vector<float>> Kernel18()
{
   int i = 0, n = 100000;
   std::vector<std::vector<float>> za(7, std::vector<float> (n+1, 0.5));
   std::vector<std::vector<float>> zp(7, std::vector<float> (n+1, 1.5));
   std::vector<std::vector<float>> zq(7, std::vector<float> (n+1, 0.25));
   std::vector<std::vector<float>> zr(7, std::vector<float> (n+1, 2.25));
   std::vector<std::vector<float>> zm(7, std::vector<float> (n+1, 0.125));
   std::vector<std::vector<float>> zb(7, std::vector<float> (n+1, 5.25));
   std::vector<std::vector<float>> zu(7, std::vector<float> (n+1, 3.25));
   std::vector<std::vector<float>> zz(7, std::vector<float> (n+1, 4.5));
   std::vector<std::vector<float>> zv(7, std::vector<float> (n+1, 0.25));
   do{
      float t = 0.0037;
      float s = 0.0041;
      int kn = 6;
      int jn = n;
      #pragma omp parallel for collapse(2) 
      for ( int k=1 ; k<kn ; k++ ) {
   /* #pragma nohazard */
         for ( int j=1 ; j<jn ; j++ ) {
             za[k][j] = ( zp[k+1][j-1] +zq[k+1][j-1] -zp[k][j-1] -zq[k][j-1] )*
                        ( zr[k][j] +zr[k][j-1] ) / ( zm[k][j-1] +zm[k+1][j-1]);
             zb[k][j] = ( zp[k][j-1] +zq[k][j-1] -zp[k][j] -zq[k][j] ) *
                        ( zr[k][j] +zr[k-1][j] ) / ( zm[k][j] +zm[k][j-1]);
         }
       }
      #pragma omp parallel for collapse(2) 
       for ( int k=1 ; k<kn ; k++ ) {
   /* #pragma nohazard */
           for ( int j=1 ; j<jn ; j++ ) {
               zu[k][j] += s*( za[k][j]   *( zz[k][j] - zz[k][j+1] ) -
                               za[k][j-1] *( zz[k][j] - zz[k][j-1] ) -
                               zb[k][j]   *( zz[k][j] - zz[k-1][j] ) +
                               zb[k+1][j] *( zz[k][j] - zz[k+1][j] ) );
               zv[k][j] += s*( za[k][j]   *( zr[k][j] - zr[k][j+1] ) -
                               za[k][j-1] *( zr[k][j] - zr[k][j-1] ) -
                               zb[k][j]   *( zr[k][j] - zr[k-1][j] ) +
                               zb[k+1][j] *( zr[k][j] - zr[k+1][j] ) );
           }
       }
      #pragma omp parallel for collapse(2) 
       for ( int k=1 ; k<kn ; k++ ) {
   /* #pragma nohazard */
           for ( int j=1 ; j<jn ; j++ ) {
               zr[k][j] = zr[k][j] + t*zu[k][j];
               zz[k][j] = zz[k][j] + t*zv[k][j];
              }
       }
   } while( i++ < iter );
   return zz;
}


/*
 *******************************************************************
 *   Kernel 19 -- general linear recurrence equations
 *******************************************************************
 *               KB5I= 0
 *           DO 194 L= 1,Loop
 *           DO 191 k= 1,n
 *         B5(k+KB5I)= SA(k) +STB5*SB(k)
 *               STB5= B5(k+KB5I) -STB5
 *191        CONTINUE
 *192        DO 193 i= 1,n
 *                  k= n-i+1
 *         B5(k+KB5I)= SA(k) +STB5*SB(k)
 *               STB5= B5(k+KB5I) -STB5
 *193        CONTINUE
 *194 CONTINUE
 */
void Kernel19()
{
    int ii = 0, n = 100000;
    int kb5i = n;
    float stb5 = 0.0015;
    std::vector<float> b5(n+kb5i, 0);
    std::vector<float> sa(n, 0);
    std::vector<float> sb(n, 0);
    InitializeLoop(b5, 0.0001);
    InitializeLoop(sa, 0.00023);
    InitializeLoop(sb, 0.00016);
    do{
      //The following two parallel optimizations won't work
      //as we keep using stb5 calculated from previous iteration
      //so can't process them simultaneously.
        #pragma omp parallel for
        for (int k = 0; k < n; k++) {
           b5[k+kb5i] = sa[k] + stb5 * sb[k];
           stb5 = b5[k+kb5i] - stb5;
        }
       #pragma omp parallel for
        for (int i = 1; i <= n; i++) {
           int k = n - i ;
           b5[k+kb5i] = sa[k] + stb5 * sb[k];
           stb5 = b5[k+kb5i] - stb5;
        } 
    } while (ii++ < 1);
    // Uncomment the function Print1DArrray could print out each array element
    // Compared to results with different threads we will find the differences
    // Print1DArray(b5);
    return;
}


/*
 *******************************************************************
 *   Kernel 20 -- Discrete ordinates transport, conditional recurrence on xx
 *******************************************************************
 *    DO 20 L= 1,Loop
 *    DO 20 k= 1,n
 *         DI= Y(k)-G(k)/( XX(k)+DK)
 *         DN= 0.2
 *         IF( DI.NE.0.0) DN= MAX( S,MIN( Z(k)/DI, T))
 *       X(k)= ((W(k)+V(k)*DN)* XX(k)+U(k))/(VX(k)+V(k)*DN)
 *    XX(k+1)= (X(k)- XX(k))*DN+ XX(k)
 * 20 CONTINUE
 */
std::vector<float> Kernel20()
{
   int i = 0, n = 10000;
   float s = 1.0, t = 0.5, dk = 1.0;
   std::vector<float> x(n, 0);
   std::vector<float> xx(n+1, 0);
   std::vector<float> y(n, 0);
   std::vector<float> g(n, 0);
   std::vector<float> z(n, 0);
   std::vector<float> u(n, 0);
   std::vector<float> v(n, 0);
   std::vector<float> w(n, 0);
   std::vector<float> vx(n, 0);
   InitializeLoop(x, 0.6);
   InitializeLoop(xx, 0.2);
   InitializeLoop(y, 0.76);
   InitializeLoop(g, 0.001);
   InitializeLoop(z, 0.00015);
   InitializeLoop(u, 0.0003);
   InitializeLoop(v, 0.008);
   InitializeLoop(w, 0.0001);
   InitializeLoop(vx, 0.0025);
    do {
        // The parallel optimization doesn't work since xx[k+1] depends on xx[k].
        // And every piece of code inside this for loop is potentially related to 
        // xx[k] so we can not even do partial optimization. 
        #pragma omp parallel for
        for (int k = 1; k < n; k++) 
        {
            float di = y[k] - g[k] / ( xx[k] + dk );
            float dn = 0.2;
            if ( di > 0) 
            {
                dn = z[k]/di ;
                if ( t < dn ) dn = t;
                if ( s > dn ) dn = s;
            }
            x[k] = ( ( w[k] + v[k]*dn )* xx[k] + u[k] ) / ( vx[k] + v[k]*dn );
            xx[k+1] = ( x[k] - xx[k] )* dn + xx[k];
        } 
    } while(++i < 1);
    // Uncomment the function Print1DArrray could print out each array element
    // Compared to results with different threads we will find the differences
    // Print1DArray(xx);
    return xx;
}

/*
 *******************************************************************
 *   Kernel 21 -- matrix*matrix product
 *******************************************************************
 *    DO 21 L= 1,Loop
 *    DO 21 k= 1,25
 *    DO 21 i= 1,25
 *    DO 21 j= 1,n
 *    PX(i,j)= PX(i,j) +VY(i,k) * CX(k,j)
 * 21 CONTINUE
 */
std::vector<std::vector<float>> Kernel21()
{
   int ii = 0, n = 10000;
   std::vector<float> x(25, 0);
   std::vector<float> y(25, 0);
   std::vector<float> z(25, 0);
   InitializeLoop(x, 0.001);
   InitializeLoop(y, 0.0023);
   InitializeLoop(z, 0.016);
   std::vector<std::vector<float>> px(n,x);
   std::vector<std::vector<float>> vy(25,y);
   std::vector<std::vector<float>> cx(n,z);
  do{ 
      for ( int k=0 ; k<25 ; k++ ) {
           #pragma omp parallel for collapse(2)
           for ( int i=0 ; i<25 ; i++ ) {
/* #pragma nohazard */
               for ( int j=0 ; j<n ; j++ ) {
                   px[j][i] += vy[k][i] * cx[j][k];
               }
           }
       }
   }while(ii++ < iter);
    // Print2DArray(px);
   return px;
}

/*
 *******************************************************************
 *   Kernel 22 -- Planckian distribution
 *******************************************************************
 *     EXPMAX= 20.0
 *       U(n)= 0.99*EXPMAX*V(n)
 *    DO 22 L= 1,Loop
 *    DO 22 k= 1,n
 *                                          Y(k)= U(k)/V(k)
 *       W(k)= X(k)/( EXP( Y(k)) -1.0)
 * 22 CONTINUE
 */

std::vector<float> Kernel22()
{
   int i = 0, n = 1000000;
   std::vector<float> u(n, 0);
   std::vector<float> v(n, 0);
   std::vector<float> w(n, 0);
   std::vector<float> x(n, 0);
   std::vector<float> y(n, 0);
   InitializeLoop(u, 0.003);
   InitializeLoop(v, 0.01);
   InitializeLoop(w, 0.006);
   InitializeLoop(x, 0.0099);
   InitializeLoop(y, 0.01);
   double expmax = 20.0;
   u[n-1] = 0.99 * expmax * v[n-1];
   do {
      #pragma omp parallel for
      for ( int k=0 ; k<n ; k++ ) {
           y[k] = u[k] / v[k];
           w[k] = x[k] / ( exp( y[k] ) -1.0 );
       }
   } while( i++ < iter );
   return w;
}

/*
 *******************************************************************
 *   Kernel 23 -- 2-D implicit hydrodynamics fragment
 *******************************************************************
 *    DO 23  L= 1,Loop
 *    DO 23  j= 2,6
 *    DO 23  k= 2,n
 *          QA= ZA(k,j+1)*ZR(k,j) +ZA(k,j-1)*ZB(k,j) +
 *   .          ZA(k+1,j)*ZU(k,j) +ZA(k-1,j)*ZV(k,j) +ZZ(k,j)
 * 23  ZA(k,j)= ZA(k,j) +.175*(QA -ZA(k,j))
 */
std::vector<std::vector<float>> Kernel23()
{
   int i = 0, n = 100000;
   std::vector<std::vector<float>> za(7, std::vector<float>(n+1, 1.2));
   std::vector<std::vector<float>> zb(7, std::vector<float>(n+1, 1.7));
   std::vector<std::vector<float>> zr(7, std::vector<float>(n+1, 2.2));
   std::vector<std::vector<float>> zu(7, std::vector<float>(n+1, 3.6));
   std::vector<std::vector<float>> zv(7, std::vector<float>(n+1, 4.2));
   std::vector<std::vector<float>> zz(7, std::vector<float>(n+1, 0.6));
   do{
      //The following parallel optimization is faster but leads to incorrect
      //results since for each iteration, the calculation depends on the values
      //calculated by previous iterations.
      #pragma omp parallel for collapse(2)
      for(int j = 1; j<6; j++) {
         for( int k=1; k<n; ++k) {
            int qa = za[j+1][k]*zr[j][k] + za[j-1][k]*zb[j][k] +
               za[j][k+1]*zu[j][k] + za[j][k-1]*zv[j][k] + zz[j][k];
            za[j][k] += 0.175*( qa - za[j][k] );
         }
      }
   } while(i++ <iter);
   return za;
}

/*
*******************************************************************
*   Kernel 24 -- find location of first minimum in array
*******************************************************************
*     X( n/2)= -1.0E+10
*    DO 24  L= 1,Loop
*           m= 1
*    DO 24  k= 2,n
*          IF( X(k).LT.X(m))  m= k
* 24 CONTINUE
*/

float Kernel24()
{
    int i = 0,n = 1000000;
    double res = 0.0;
    std::vector<float> x(n ,1.5);
    InitializeLoop(x, 0.001);
    x[n/2] = -1.0e+10;

    const int num_threads = omp_get_max_threads();
    std::vector<int> m(num_threads,0); // stores the minimal in each thread
    do {
        #pragma omp parallel for
        for(int k = 0; k < n; k++){
            int thread = omp_get_thread_num();
            if(x[k] < x[m[thread]]){
                m[thread] = k;
            }
        }
        //Find the minimal among all threads
        res = x[m[0]];
        for(int k = 0; k < num_threads; ++k){
            if(res > x[m[k]])
                res = x[m[k]];
        }

        //The following code is faster but might have data race
        /*
        int m = 0;
        #pragma omp parallel for
        for( int k = 1; k<n; k++){
            if( x[k] < x[m]) {
               #pragma omp atomic write
               m = k;
            }
        }        
        res = x[m]; */

    } while(i++ < iter);
    return res;
}


int main(int argc, char *argv[])
{
    if( argc >= 3 )
    {
        // Choose which kernel loop that we want to run
        const int kernel_test = atoi( argv[ 1 ] );
        std::cout << "The kernel we used for testing is: " << kernel_test << std::endl;
        // Set the number of cores that we want to use
        omp_set_num_threads( atoi( argv[ 2 ] ) );
        int threads_num = omp_get_max_threads();
        std::cout << "The threads number we used for testing is " << threads_num << std::endl;

        // Create a variable sum to calculate the sum of each elements in the array in each kernel
        // In order to make sure the result is correct
        long double sum = 0;

        // Start to recored the time
        auto const start_time = std::chrono::steady_clock::now();

        switch(kernel_test)
        {
        case 1:
        {   
            auto k1 = Kernel1();
            sum = CalculateSum(k1);
            break;
        }
        case 2:
        {
            auto k2 = Kernel2();
            sum = CalculateSum(k2);
            break;
        } 
        case 3:
        {
            sum = Kernel3();
            break;
        } 
        case 4:
        {
            auto k4 = Kernel4();
            sum = CalculateSum(k4);
            break;
        } 
        case 5:
        {
            auto k5 = Kernel5();
            sum = CalculateSum(k5);
            break;
        } 
        case 6:
        {
            auto k6 = Kernel6();
            sum = CalculateSum(k6);
            break;
        } 
        case 7:
        {
            auto k7 = Kernel7();
            sum = CalculateSum(k7);
            break;
        } 
        case 8:
        {
            Kernel8();
            break;
        } 
        case 9:
        {
            Kernel9();
            // Could use Print2DArray() to print the results of the array to compare.
            break;
        } 
        case 10:
        {
            Kernel10();
            // Could use Print2DArray() to print the results of the array to compare.
            break;
        }
        case 11:
        {
            auto k11 = Kernel11();
            sum = CalculateSum(k11);
            break;
        }
        case 12:
        {
            auto k12 = Kernel12();
            sum = CalculateSum(k12);
            break;
        } 
        case 13:
        {
            Kernel13();
            // Could use Print2DArray() to print the results of the array to compare.
            break;
        }
        case 14:
        {
            Kernel14();
            break;
        }
        case 15:
        {
            std::vector<std::vector<float>> k15 = Kernel15();
            break;
        }
        case 16:
        {
            auto k16 = Kernel16();
            break;
        }
        case 17:
        {
            auto k17 = Kernel17();
            //printVector(k17);
            break;
        }
        case 18:
        {
            std::vector<std::vector<float>> k18 = Kernel18();
            //`print2dVector(k18);
            break;
        }
        case 19:
        {
           Kernel19();
           break;
        }
        case 20:
        {
           auto k20 = Kernel20();
           sum = CalculateSum(k20);
           break;
        }
        case 21:
        {
            std::vector<std::vector<float>> k21 = Kernel21();
            break;
        }
        case 22:
        {
            auto k22 = Kernel22();
            // Verified the parallel version generates same
            // results as baseline by printing out all elements
            // in the vector.
            break;
        }
        case 23:
       {
            std::vector<std::vector<float>> k23 = Kernel23();
            break;
        }
        case 24:
        {
            auto k24 = Kernel24();
            sum = k24;
            break;
        }
        default:
            std::cout << "Invalid Input!" << std::endl;
            std::cout << "The sencond parameter(kernel number) should be less than 24" << std::endl;
            std::cout << "The third parameter(threads number) should be less than the threads you have" << std::endl;
        }      
        
        auto const end_time = std::chrono::steady_clock::now();
        auto const avg_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "Total execution time  = " << avg_time << " us" << std::endl;
        std::cout << "Total sum for the kernel array is: " << sum << std::endl;
        
    }
    else
    {
        std::cout << "Invalid Input: " << std::endl;
        std::cout << "Please input the kernel loops wanted to test in argc[1]" << std::endl;
        std::cout << "Please input the number of threads wanted to use in argc[2]" << std::endl;
    }
    
    return 0;
}



