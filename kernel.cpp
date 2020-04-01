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
        #pragma omp parallel for
        for (int k = 6; k < 10000; k = k + m) 
        {
            int lw = k - 6;
            float temp = x[k-1];
            /* #pragma nohazard */
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
// std::vector<float> Kernel8()
// {
//     int i = 0, n = 1000;
//     float du1[n] = {0.1};
//     int u1[2][1000][4] = {3, 4, 2, 3, 0, -3, 9, 11, 23, 12, 23, 
//                  2, 13, 4, 56, 3, 5, 9, 3, 5, 5, 1, 4, 9};

//     do {
//     int nl1 = 0;
//     int nl2 = 1;
//     for (int kx = 1; kx < 3; kx++)
//     {
//        for (int ky = 1; ky < n; ky++) 
//        {
//             du1[ky] = u1[nl1][ky+1][kx] - u1[nl1][ky-1][kx];
//             du2[ky] = u2[nl1][ky+1][kx] - u2[nl1][ky-1][kx];
//             du3[ky] = u3[nl1][ky+1][kx] - u3[nl1][ky-1][kx];
//             u1[nl2][ky][kx] = u1[nl1][ky][kx] + a11 * du1[ky] + a12 * du2[ky] + a13 * du3[ky] + sig *
//                 (u1[nl1][ky][kx+1] - 2.0 * u1[nl1][ky][kx] + u1[nl1][ky][kx-1]);
//             u2[nl2][ky][kx] = u2[nl1][ky][kx] + a21 * du1[ky] + a22 * du2[ky] + a23 * du3[ky] + sig *
//                 (u2[nl1][ky][kx+1] - 2.0 * u2[nl1][ky][kx] + u2[nl1][ky][kx-1]);
//             u3[nl2][ky][kx] = u3[nl1][ky][kx] + a31 * du1[ky] + a32 * du2[ky] + a33 * du3[ky] + sig *
//                 (u3[nl1][ky][kx+1] - 2.0 * u3[nl1][ky][kx] + u3[nl1][ky][kx-1]);
//        }
//     }
//    } while(i++ < iter);
// }

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
    float ar = 0.05, br = 0.02, cr = 0.012;
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
    } while( ++j < 1 );
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
        // case 8:
        // {
        //     auto k8 = Kernel8();
        //     sum = CalculateSum(k8);
        //     break;
        // } 
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



