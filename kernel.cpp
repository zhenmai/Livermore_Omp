#include <iostream>
#include <cstring>
#include <string>
#include <cmath>
#include <vector>
#include <chrono>
#include <ctime>
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
    for (int i = 0; i < input.size(); i++)
    {
        input[i] = i * mul;
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
    float q = 0.5, r = 0.2, t = 0.1;
    std::vector<float> x(n, 0);
    std::vector<float> y(n, 1.3);
    std::vector<float> z(n + 11, 3.5);
    do {
        #pragma omp parallel for
        for (int k = 0; k < n; k++)
        {
            x[k] = q + y[k] * (r * z[k+10] + t * z[k+11]);
        }
    } while( i++ < iter);
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
    int ipntp, ipnt, i, ii, n = 5000, j = 0;
    std::vector<float> x(n, 0);
    std::vector<float> v(n, 0);
    InitializeLoop(x, 0.001);
    InitializeLoop(v, 0.0029);
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
            /* #pragma nohazard */
            #pragma omp parallel for
            for (int k = ipnt+1; k< ipntp ; k = k+2 ) 
            {
                i++;
                x[i] = x[k] - v[k] * x[k-1] - v[k+1] * x[k+1];
            }
        } while ( ii>0 );
    } while( j++ < iter );
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
    int i = 0, n = 10000;
    std::vector<float> z(n, 0.2);
    std::vector<float> x(n, 0.06);
    do 
    {
       #pragma omp parallel for reduction(+:q)
       for (int k = 0; k < n; k++) 
       {
           q += z[k] * x[k];    
       }
    } while(i++ < iter);
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
    int j = 0, n = 100000;
    int m = (1001 - 7)/2;
    std::vector<float> x(n, 0);
    std::vector<float> y(n, 0);
    InitializeLoop(x, 0.001);
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
    } while(j++ < iter);
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
    int j = 0, n = 1000;
    std::vector<float> x(n, 1.2);
    std::vector<float> y(n, 3.6);
    std::vector<float> z(n, 0.7);
    
    do {
        // This loop can't be parallelize because the loop dependence
        for (int i = 1; i < n; i++ ) 
        {
           x[i] = z[i] * (y[i] - x[i-1]);
        }
    } while(j++ < iter);
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

// Haven't finished, to see weather that could be parallelized
std::vector<float> Kernel6()
{
    int j = 0, n = 1000;
    std::vector<float> w(n, 1);
    InitializeLoop(w, 0.001);
    std::vector<std::vector<float> > b(n, std::vector<float>(n, 0.067));
    do {
        #pragma omp parallel for
        for (int i = 1; i < n; i++) 
        {
            w[i] = 0.0100;
            for (int k = 0; k < i; k++) 
            {
                w[i] += b[k][i] * w[(i-k)-1];
            }
        } 
    } while(j++ < iter);
    return w;
}



int main(int argc, char *argv[])
{
    if( argc >= 3 )
    {
        // Choose which kernel loop that we want to run
        const int kernel_test = atoi( argv[ 1 ] );
        std::cout << "The kernel we are testing is: " << kernel_test << std::endl;
        // Set the number of cores that we want to use
        omp_set_num_threads( atoi( argv[ 2 ] ) );
        int threads_num = omp_get_max_threads();
        printf("The used threads number is: %d \n", threads_num);

        // Create a variable sum to calculate the sum of each elements in the array 
        // In order to make sure the result is correct
        double sum = 0;

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
            auto k2 = Kernel1();
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
        default:
            std::cout << "Invalid Input!" << std::endl;
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



