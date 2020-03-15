#include <iostream>
#include <cstring>
#include <string>
#include <cmath>
#include <vector>
#include <chrono>
#include <ctime>
#include <omp.h>

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
        ++i;
    } while( i < 100);
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
    int ipntp, ipnt, i, ii, n = 1000, j = 0;
    std::vector<float> x(n, 0);
    std::vector<float> v(n, 0);
    do {
        ii = n;
        ipntp = 0;
        do {
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
        j++;
    } while( j < 1000 );
    return x;
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
            // Calculate the time used for kernel 2
            auto k2 = Kernel1();
            sum = CalculateSum(k2);
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



