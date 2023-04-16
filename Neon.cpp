#include <arm_neon.h> // Neon
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <time.h>
#include<chrono>

using namespace std;
using namespace chrono;

#define PI 3.1415926535897932384626433832


struct COMPLEX
{
    double real = 0;
    double imag = 0;
};


template<typename T>
void separate(T* a, int n)
{
    T* b = new T[n / 2];				// get temp heap storage
    for (int i = 0; i < n / 2; i++)		// copy all odd elements to heap storage
        b[i] = a[i * 2 + 1];
    for (int i = 0; i < n / 2; i++)		// copy all even elements to lower-half of a[]
        a[i] = a[i * 2];
    for (int i = 0; i < n / 2; i++)		// copy all odd (from heap) to upper-half of a[]
        a[i + n / 2] = b[i];
    delete[] b;							// delete heap storage
}



// 串行dft算法
void dft(vector<double> input, vector<complex<double>>& output)
{
    size_t length = input.size();
    for (size_t i = 0; i < length; i++)
    {
        double real = 0;
        double imag = 0;
        for (size_t j = 0; j < length; j++)
        {
            // X[k] += x[n] * exp( (double)k *-1i * 2.0*PI*
            real += input[j] * cos(-double(i * 2 * PI * j / length));
            imag += input[j] * sin(-double(i * 2 * PI * j / length));
        }
        output[i] = complex<double>(real, imag);
    }
}


// 串行fft算法
void fft(vector<double> input, vector<complex<double>>& output)
{
    
    size_t length = input.size();
    if (length >= 2)
    {
        // 分为奇偶
        vector<double> odd;
        vector<double> even;
        for (size_t n = 0; n < length; n++)
        {
            if (n & 1)
            {
                odd.push_back(input.at(n));
            }
            else
            {
                even.push_back(input.at(n));
            }

        }

        // 重排
        // 低
        vector<complex<double>> fft_even_out(output.begin(), output.begin() + length / 2);
        // 高
        vector<complex<double>> fft_odd_out(output.begin() + length / 2, output.end());
        // 递归执行代码
        fft(even, fft_even_out);
        fft(odd, fft_odd_out);

        // 组合奇偶部分
        complex<double> odd_out;
        complex<double> even_out;
        for (size_t n = 0; n != length / 2; n++)
        {
            if (length == 2)
            {
                even_out = even[n] + fft_even_out[n];
                odd_out = odd[n] + fft_odd_out[n];
            }
            else
            {
                even_out = fft_even_out[n];
                odd_out = fft_odd_out[n];
            }
            // 翻转因子
            complex<double> w = exp(complex<double>(0, -2.0 * PI * double(n) / (double)(length)));
            // even part
            output[n] = even_out + w * odd_out;
            // odd part
            output[n + length / 2] = even_out - w * odd_out;
        }
    }
}


void fft_Neon(COMPLEX* X, int N) 
{
    if (N < 2) 
    {
        // bottom of recursion.
        // Do nothing here, because already X[0] = x[0]
    }
    else 
    {
        if (N < 2) 
        {
            // bottom of recursion.
            // Do nothing here, because already X[0] = x[0]
        }
        else 
        {
            separate(X, N);                        // ż��������ǰ�������ں�
            fft_Neon(X, N / 2);
            fft_Neon(X + N / 2, N / 2);

            for (int k = 0; k < N / 2; k++) 
            {
                //����Ϊһ���˼�����
                double coss = cos(-2. * PI * k / N);
                double sinn = sin(-2. * PI * k / N);

                float64x2_t even1, odd1;
                //��������˷����
                float64x2_t cos_2vec = { coss, coss };
                float64x2_t sin_2vec = { sinn, sinn };
                float64x2_t zero = { 0.0, 0.0 };

                // ��������
                double* ptr1 = (double*)&X[k];
                double* ptr2 = (double*)&X[k + N / 2];
                // ����ָ�벢��������
                even1 = vld1q_f64(ptr1);
                odd1 = vld1q_f64(ptr2);

                //����ÿһ�����㣨e+w*o)���ȼ���˷�w*o
                //����˷�
                float64x2_t intew1 = vmulq_f64(cos_2vec, odd1);//�������ac��bc
                float64x2_t intew2 = vmulq_f64(sin_2vec, odd1);//�������ad��bd

                //�������
                float64x2_t intew2_neg = vnegq_f64(intew2);

                //ʵ��ac-bd  �鲿ad+bc
                //����˳����Ҫ����˳��ʹ��λ���
                float64x2_t sf_intew2 = vextq_f64(intew2, intew2_neg, 1);

                float64x2_t real_o = vaddq_f64(intew1, sf_intew2);

                //����ӷ�
                float64x2_t result_e = vaddq_f64(even1, real_o);
                float64x2_t _result_o = vsubq_f64(even1, real_o);

                //������
                vst1q_f64((double*)&X[k], result_e);
                vst1q_f64((double*)&X[k + N / 2], _result_o);
            }
        } 
    }
}



// 结果
vector<complex<double>> result(1024);
// 改数据规模时要改的
COMPLEX* input = new COMPLEX[1024];
// 测试函数，每个算法循环测试counter轮
void test(vector<double>& data)
{
    
    // 循环测试次数100
    size_t counter = 100;
    // 数据规模
    size_t size = data.size();

    // dft
    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();// 计时开始
    for (size_t n = 0; n < counter; n++)
    {
        dft(data, result);
    }
    auto t2 = Clock::now();// 计时结束
    cout << "dft cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 << " ms.\n";
    // fft 串行

    t1 = Clock::now();// 计时开始
    for (size_t n = 0; n < counter; n++)
    {
        fft(data, result);
    }
    t2 = Clock::now();// 计时结束

    cout << "fft 串行 cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 << " ms.\n";
    // fft Neon
    // load data
    for (size_t i = 0; i < size; i++)
    {
        input[i].real = data[i];
        input[i].imag = 0;
    }
    t1 = Clock::now();// 计时开始
    for (size_t n = 0; n < counter; n++)
    {
        fft_Neon(input, size);
    }
    t2 = Clock::now();// 计时结束
    cout << "fft_Neon cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 << " ms.\n";
}


int main() {
    // 读取数据集
    ifstream fi("/home/ss2113384/fft_1024.txt");
    vector<double> data;
    string read_temp;
    while (fi.good())
    {
        getline(fi, read_temp);
        data.push_back(stod(read_temp));
    }

    test(data);

    // 验证结果的正确性
    ofstream fo;
    COMPLEX* input = new COMPLEX[1024];
    for (size_t i = 0; i < 1024; i++)
    {
        input[i].real = data[i];
        input[i].imag = 0;
    }
    fft_Neon(input, 1024);
    for (int i = 0; i < 1024; i++)
    {
        cout << '(' << input[i].real << ',' << input[i].imag << ')' << endl;
    }
    fi.close();

    return 0;
}

