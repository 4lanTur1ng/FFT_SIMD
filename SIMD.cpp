#include <iostream>
#include <vector>
#include <immintrin.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <time.h>
#include <windows.h>


using namespace std;
#define PI 3.1415926535897932384626433832


struct COMPLEX
{
	double real = 0;
	double imag = 0;
};

void dft(vector<double> input, vector<complex<double>> &output)
{
	// 串行dft算法
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
		output[i]=complex<double>(real, imag);
	}
}


template<typename T>
void separate(T* a, int n)
{
	if (n < 2)
	{
		return;
	}
	else
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
}


// 对划分进行simd优化
void separate_simd(COMPLEX* input, size_t length)
{
	if (length < 2)
	{
		return;
	}
	else
	{
		COMPLEX* b = new COMPLEX[length / 2];
		for (int i = 0; i < length / 2; i++)
		{
			__m128d temp = _mm_load_pd((double*)&input[i*2+1]);
			_mm_store_pd((double*)&b[i], temp);
		}
		for (int i = 0; i < length / 2; i++)
		{
			__m128d temp = _mm_load_pd((double*)&input[i* 2]);
			_mm_store_pd((double*)&input[i], temp);
		}
		for (int i = 0; i < length / 2; i++)
		{
			__m128d temp = _mm_load_pd((double*)&b[i]);
			_mm_store_pd((double*)&input[i + length / 2], temp);
		}


		delete[] b;
	}

}


void fft(vector<double> input, vector<complex<double>> &output)
{
	// 串行fft算法
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


void fft_sse(COMPLEX* input, size_t length)
{
	// size_t length = sizeof(input) / sizeof(input[0]);;
	if (length >= 2)
	{
		// 重排
		separate(input, length);

		// 递归执行代码
		fft_sse(input, length / 2);
		fft_sse(input + length / 2, length / 2);

		for (size_t i = 0; i < length / 2; i++)
		{
			// odd
			__m128d o = _mm_loadu_pd((double*)&input[i + length / 2]);
			// 旋转因子
			double COS = cos(-2. * PI * i / length);
			double SIN = sin(-2. * PI * i / length);

			__m128d wr = _mm_set1_pd(cos(-2. * PI * i / length));
			__m128d wi = _mm_set1_pd(sin(-2. * PI * i / length));
			
			// r*cos i*cos
			wr = _mm_mul_pd(o, wr);
			// 将向量o中的元素进行交换
			__m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1));
			wi = _mm_mul_pd(n1, wi);				// bd|ad
			n1 = _mm_sub_pd(wr, wi);				// ac-bd|x
			wr = _mm_add_pd(wr, wi);				// x|bc+ad
			n1 = _mm_shuffle_pd(n1, wr, _MM_SHUFFLE2(1, 0));	// select ac-bd|bc+ad
			o = _mm_loadu_pd((double*)&input[i]);
			wr = _mm_add_pd(o, n1);
			wi = _mm_sub_pd(o, n1);
			_mm_storeu_pd((double*)&input[i], wr);
			_mm_storeu_pd((double*)&input[i + length / 2], wi);
		}
	}
	else
	{
		// do nothing
	}

}


void fft_sse_s(COMPLEX* input, size_t length)
{
	// size_t length = sizeof(input) / sizeof(input[0]);;
	if (length >= 2)
	{
		// 重排
		separate_simd(input, length);

		// 递归执行代码
		fft_sse_s(input, length / 2);
		fft_sse_s(input + length / 2, length / 2);

		for (size_t i = 0; i < length / 2; i++)
		{
			__m128d o = _mm_load_pd((double*)&input[i + length / 2]);   // odd a|b
			double SIN = sin(-2. * PI * i / length);

			__m128d wr = _mm_set1_pd(cos(-2. * PI * i / length));			//__m128d wr =  _mm_set_pd( cc,cc );		// cc 
			__m128d wi = _mm_set_pd(SIN, -SIN);		// -d | d	, note that it is reverse order
			// compute the w*o
			wr = _mm_mul_pd(o, wr);					// ac|bc
			__m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1)); // invert  bc|ac
			wi = _mm_mul_pd(n1, wi);				// -bd|ad
			n1 = _mm_add_pd(wr, wi);				// ac-bd|bc+ad 

			o = _mm_load_pd((double*)&input[i]);	// load even part
			wr = _mm_add_pd(o, n1);					// compute even part, X_e + w * X_o;
			wi = _mm_sub_pd(o, n1);					// compute odd part,  X_e - w * X_o;
			_mm_store_pd((double*)&input[i], wr);
			_mm_store_pd((double*)&input[i + length / 2], wi);
		}
	}
	else
	{
		// do nothing
	}

}


void fft_sse2(COMPLEX* input, size_t length)
{
	if (length >= 2)
	{
		separate(input, length);
		fft_sse2(input, length / 2);
		fft_sse2(input + length / 2, length / 2);

		for (size_t i = 0; i < length / 2; i++) {

			__m128d o = _mm_load_pd((double*)&input[i + length / 2]);   // odd a|b
			double SIN = sin(-2. * PI * i / length);

			__m128d wr = _mm_set1_pd(cos(-2. * PI * i / length));			//__m128d wr =  _mm_set_pd( cc,cc );		// cc 
			__m128d wi = _mm_set_pd(SIN, -SIN);		// -d | d	, note that it is reverse order
			// compute the w*o
			wr = _mm_mul_pd(o, wr);					// ac|bc
			__m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1)); // invert  bc|ac
			wi = _mm_mul_pd(n1, wi);				// -bd|ad
			n1 = _mm_add_pd(wr, wi);				// ac-bd|bc+ad 

			o = _mm_load_pd((double*)&input[i]);	// load even part
			wr = _mm_add_pd(o, n1);					// compute even part, X_e + w * X_o;
			wi = _mm_sub_pd(o, n1);					// compute odd part,  X_e - w * X_o;
			_mm_store_pd((double*)&input[i], wr);
			_mm_store_pd((double*)&input[i + length / 2], wi);
		}
	}
	else
	{
		// do nothing
	}

}


void fft0_avx512(COMPLEX* input, size_t length)
{
	if (length >= 8)
	{
		// 重排
		separate(input, length);
		// 递归执行代码
		fft0_avx512(input, length / 2);
		fft0_avx512(input + length / 2, length / 2);
		for (int i = 0; i < length / 2; i += 4)
		{
			// 未优化
			__m512d o = _mm512_load_pd((double*)&input[i + length / 2]);   // odd a|b
			__m512d wr = _mm512_set_pd(
				cos(-2. * PI * (i + 3) / length), cos(-2. * PI * (i + 3) / length),
				cos(-2. * PI * (i + 2) / length), cos(-2. * PI * (i + 2) / length),
				cos(-2. * PI * (i + 1) / length), cos(-2. * PI * (i + 1) / length),
				cos(-2. * PI * (i + 0) / length), cos(-2. * PI * (i + 0) / length)
			);		//__m512d wr =  _mm_set_pd( cc,cc );		// cc 

			__m512d wi = _mm512_set_pd(
				sin(-2. * PI * (i + 3) / length), sin(-2. * PI * (i + 3) / length),
				sin(-2. * PI * (i + 2) / length), sin(-2. * PI * (i + 2) / length),
				sin(-2. * PI * (i + 1) / length), sin(-2. * PI * (i + 1) / length),
				sin(-2. * PI * (i + 0) / length), sin(-2. * PI * (i + 0) / length)
				);		// -d | d	, note that it is reverse order
			// compute the w*o
			wr = _mm512_mul_pd(o, wr);					// ac|bc
			__m512d n1 = _mm512_shuffle_pd(o, o, 0x55); // invert  bc|ac
			wi = _mm512_mul_pd(n1, wi);				// -bd|ad
			n1 = _mm512_sub_pd(wr, wi);				// ac-bd|bc+ad 

			wr = _mm512_add_pd(wr, wi);				// x|bc+ad
			n1 = _mm512_shuffle_pd(n1, wr, 0xaa);	// select ac-bd|bc+ad
			o = _mm512_load_pd((double*)&input[i]);
			wr = _mm512_add_pd(o, n1);
			wi = _mm512_sub_pd(o, n1);		// compute odd part,  x_e - w * x_o;
			_mm512_store_pd((double*)&input[i], wr);
			_mm512_store_pd((double*)&input[i + length / 2], wi);


			// 优化

			//__m512d o = _mm512_load_pd((double*)&input[i + length / 2]);   // odd a|b
			//__m512d angle = _mm512_set_pd(
			//	-2. * PI * (i + 3) / length, 2. * PI * (i + 3) / length,
			//	-2. * PI * (i + 2) / length, 2. * PI * (i + 2) / length,
			//	-2. * PI * (i + 1) / length, 2. * PI * (i + 1) / length,
			//	-2. * PI zai* i / length, 2. * PI * i / length
			//);

			//__m512d wr = _mm512_cos_pd(angle);			//__m128d wr =  _mm_set_pd( cc,cc );		// cc 
			//__m512d wi = _mm512_sin_pd(angle);		// -d | d	, note that it is reverse order
			//// compute the w*o
			//wr = _mm512_mul_pd(o, wr);					// ac|bc
			//__m512d n1 = _mm512_shuffle_pd(o, o, 0x55); // invert  bc|ac
			//wi = _mm512_mul_pd(n1, wi);				// -bd|ad
			//n1 = _mm512_add_pd(wr, wi);				// ac-bd|bc+ad 

			//o = _mm512_load_pd((double*)&input[i]);	// load even part
			//wr = _mm512_add_pd(o, n1);					// compute even part, X_e + w * X_o;
			//wi = _mm512_sub_pd(o, n1);					// compute odd part,  X_e - w * X_o;
			//_mm512_store_pd((double*)&input[i], wr);
			//_mm512_store_pd((double*)&input[i + length / 2], wi);
		}
	}
	else
	{
		// do nothing
		if (length >= 2)
		{
			separate(input, length);
			fft_sse2(input, length / 2);
			fft_sse2(input + length / 2, length / 2);

			for (int i = 0; i < length / 2; i++)
			{

				__m128d o = _mm_load_pd((double*)&input[i + length / 2]);   // odd
				__m128d wr = _mm_set1_pd(cos(-2. * PI * i / length));			//__m128d wr =  _mm_set_pd( cc,cc );		// cc 
				__m128d wi = _mm_set_pd(sin(-2. * PI * i / length), -sin(-2. * PI * i / length));		// -d | d	, note that it is reverse order
				// compute the w*o
				wr = _mm_mul_pd(o, wr);					// ac|bc
				__m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1)); // invert
				wi = _mm_mul_pd(n1, wi);				// -bd|ad
				n1 = _mm_add_pd(wr, wi);				// ac-bd|bc+ad 

				o = _mm_load_pd((double*)&input[i]);	// load even part
				wr = _mm_add_pd(o, n1);					// compute even part, x_e + w * x_o;
				wi = _mm_sub_pd(o, n1);					// compute odd part,  x_e - w * x_o;
				_mm_store_pd((double*)&input[i], wr);
				_mm_store_pd((double*)&input[i + length / 2], wi);
			}
		}
		else
		{
			return;
		}
	}

}


void fft_avx512(COMPLEX* input, size_t length)
{
	if (length >= 8)
	{
		// 重排
		separate(input, length);
		// 递归执行代码
		fft_avx512(input, length / 2);
		fft_avx512(input + length / 2, length / 2);
		for (int i = 0; i < length / 2; i += 4)
		{
			// 未优化
//
//			__m512d o = _mm512_load_pd((double*)&input[i + length / 2]);   // odd a|b
//			__m512d angle = _mm512_set_pd(
//				-2. * PI * (i + 3) / length, -2. * PI * (i + 3) / length, 
//				-2. * PI * (i + 2) / length, -2. * PI * (i + 2) / length, 
//				-2. * PI * (i + 1) / length, -2. * PI * (i + 1) / length, 
//				-2. * PI * i / length , -2. * PI * i / length
//				);
//			__m512d wr = _mm512_set_pd(
//				cos(-2. * PI * (i + 3) / length), cos(-2. * PI * (i + 3) / length),
//				cos(-2. * PI * (i + 2) / length), cos(-2. * PI * (i + 2) / length),
//				cos(-2. * PI * (i + 1) / length), cos(-2. * PI * (i + 1) / length),
//				cos(-2. * PI * (i + 0) / length), cos(-2. * PI * (i + 0) / length)
//			);	*/		//__m512d wr =  _mm_set_pd( cc,cc );		// cc 
//
//			__m512d wr = _mm512_cos_pd(angle);
//		/*	__m512d wi = _mm512_set_pd(
//				sin(-2. * PI * (i + 3) / length), sin(-2. * PI * (i + 3) / length),
//				sin(-2. * PI * (i + 2) / length), sin(-2. * PI * (i + 2) / length),
//				sin(-2. * PI * (i + 1) / length), sin(-2. * PI * (i + 1) / length),
//				sin(-2. * PI * (i + 0) / length), sin(-2. * PI * (i + 0) / length)
//				);*/		// -d | d	, note that it is reverse order
//
//			__m512d wi = _mm512_sin_pd(angle);
//			// compute the w*o
//			wr = _mm512_mul_pd(o, wr);					// ac|bc
//			__m512d n1 = _mm512_shuffle_pd(o, o, 0x55); // invert  bc|ac
//			wi = _mm512_mul_pd(n1, wi);				// -bd|ad
//			n1 = _mm512_sub_pd(wr, wi);				// ac-bd|bc+ad 
//
//			wr = _mm512_add_pd(wr, wi);				// x|bc+ad
//			n1 = _mm512_shuffle_pd(n1, wr, 0xaa);	// select ac-bd|bc+ad
//			o = _mm512_load_pd((double*)&input[i]);
//			wr = _mm512_add_pd(o, n1);
//			wi = _mm512_sub_pd(o, n1);		// compute odd part,  x_e - w * x_o;
//			_mm512_store_pd((double*)&input[i], wr);
//			_mm512_store_pd((double*)&input[i + length / 2], wi);
//

			// 优化

			__m512d o = _mm512_load_pd((double*)&input[i + length / 2]);   // odd a|b
			__m512d angle = _mm512_set_pd(
				-2. * PI * (i + 3) / length, 2. * PI * (i + 3) / length,
				-2. * PI * (i + 2) / length, 2. * PI * (i + 2) / length,
				-2. * PI * (i + 1) / length, 2. * PI * (i + 1) / length,
				-2. * PI * i / length, 2. * PI * i / length
			);

			__m512d wr = _mm512_cos_pd(angle);			//__m128d wr =  _mm_set_pd( cc,cc );		// cc 
			__m512d wi = _mm512_sin_pd(angle);		// -d | d	, note that it is reverse order
			// compute the w*o
			wr = _mm512_mul_pd(o, wr);					// ac|bc
			__m512d n1 = _mm512_shuffle_pd(o, o, 0x55); // invert  bc|ac
			wi = _mm512_mul_pd(n1, wi);				// -bd|ad
			n1 = _mm512_add_pd(wr, wi);				// ac-bd|bc+ad 

			o = _mm512_load_pd((double*)&input[i]);	// load even part
			wr = _mm512_add_pd(o, n1);					// compute even part, X_e + w * X_o;
			wi = _mm512_sub_pd(o, n1);					// compute odd part,  X_e - w * X_o;
			_mm512_store_pd((double*)&input[i], wr);
			_mm512_store_pd((double*)&input[i + length / 2], wi);
		}
	}
	else
	{
		// do nothing
		if (length >= 2)
		{
			separate(input, length);
			fft_sse2(input, length / 2);
			fft_sse2(input + length / 2, length / 2);

			for (int i = 0; i < length / 2; i++) 
			{

				__m128d o = _mm_load_pd((double*)&input[i + length / 2]);   // odd
				__m128d wr = _mm_set1_pd(cos(-2. * PI * i / length));			//__m128d wr =  _mm_set_pd( cc,cc );		// cc 
				__m128d wi = _mm_set_pd(sin(-2. * PI * i / length), -sin(-2. * PI * i / length));		// -d | d	, note that it is reverse order
				// compute the w*o
				wr = _mm_mul_pd(o, wr);					// ac|bc
				__m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1)); // invert
				wi = _mm_mul_pd(n1, wi);				// -bd|ad
				n1 = _mm_add_pd(wr, wi);				// ac-bd|bc+ad 

				o = _mm_load_pd((double*)&input[i]);	// load even part
				wr = _mm_add_pd(o, n1);					// compute even part, x_e + w * x_o;
				wi = _mm_sub_pd(o, n1);					// compute odd part,  x_e - w * x_o;
				_mm_store_pd((double*)&input[i], wr);
				_mm_store_pd((double*)&input[i + length / 2], wi);
			}
		}
		else
		{
			return;
		}
	}
}


COMPLEX *input = new COMPLEX[8192];
COMPLEX* input_avx = new COMPLEX[8192];
vector<complex<double>> result(8192);	// 结果


void test(vector<double> &data)
{
	// 测试函数，每个算法循环测试counter
	// 循环测试次数100
	size_t counter = 1000;
	size_t size = data.size();
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	double quadpart = (double)frequency.QuadPart;
	// dft
	LARGE_INTEGER start;
	LARGE_INTEGER end;
	//QueryPerformanceCounter(&start);
	//for (size_t n = 0; n < counter; n++) 
	//{
	//	fft(data, result);
	//}
	//QueryPerformanceCounter(&end);
	//cout << "fft cost : " << (end.QuadPart - start.QuadPart) / quadpart * 1000 / counter << " ms.\n";
	// fft 串行
	//QueryPerformanceCounter(&start);

	//for (size_t n = 0; n < counter; n++) 
	//{
	//	fft(data, result);
	//}
	//QueryPerformanceCounter(&end);

	//cout << "fft 串行 cost : " << (end.QuadPart - start.QuadPart) / quadpart * 1000/ counter << " ms.\n";

	// fft SIMD SSE
	// load data

	for (size_t i = 0; i < size; i++)
	{
		input[i].real = data[i];
		input[i].imag = 0;
	}
	
	QueryPerformanceCounter(&start);

	for (size_t n = 0; n < counter; n++)
	{
		fft_sse(input, size);
	}

	QueryPerformanceCounter(&end);
	cout << "fft_sse cost : " << (end.QuadPart - start.QuadPart) / quadpart /counter * 1000 << " ms.\n";
}


int main(){
	// 读取数据集
	ifstream fi("fft_8192.txt");
	vector<double> data;
	string read_temp;
	while (fi.good())
	{
		getline(fi, read_temp);
		data.push_back(stod(read_temp));
	}

	test(data);
	
	// 验证结果的正确性
	//ofstream fo;
	//vector<complex<double>> result(1024);
	//fft(data, result);
	//fo.open("fft_result.txt", ios::out);
	//for (int i = 0; i < result.size(); i++)
	//{
	//	fo << result[i] << endl;
	//}
	//COMPLEX *input = new COMPLEX[1024];
	//for (size_t i = 0; i < 1024; i++)
	//{
	//	input[i].real = data[i];
	//	input[i].imag = 0;
	//}
	//fft_sse_s(input, 1024);
	//fo.open("fft_avx1024_result.txt", ios::out);
	//for (int i = 0; i < 1024; i++)
	//{
	//	fo <<'(' << input[i].real<<','<< input[i].imag << ')' << endl;
	//}
	//fo.close();

	fi.close();


	return 0;
}
