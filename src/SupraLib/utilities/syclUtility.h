// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2016, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __CUDAUTILITY_H__
#define __CUDAUTILITY_H__

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#ifdef HAVE_CUDA
#ifdef HAVE_CUFFT
#include <cufft.h>
#endif
#endif
#include <cstdio>
#include "utilities/Logging.h"
#include <algorithm>
#include <cmath>
#include <cfloat>

namespace supra
{
#ifdef SYCL_LANGUAGE_VERSION
	using sycl::max;
	using sycl::min;
	using sycl::round;
	using sycl::floor;
	using sycl::ceil;
#else
	using std::max;
	using std::min;
	using std::round;
	using std::floor;
	using std::ceil;
#endif

#ifdef HAVE_CUDA
	//define for portable function name resolution
	#if defined(__GNUC__)
	//GCC
	/// Name of the function this define is referenced. GCC version
	#define FUNCNAME_PORTABLE __PRETTY_FUNCTION__
	#elif defined(_MSC_VER)
	//Visual Studio
	/// Name of the function this define is referenced. Visual Studio version
	#define FUNCNAME_PORTABLE __FUNCSIG__
	#endif

	/// Verifies a cuda call returned "cudaSuccess". Prints error message otherwise.
	/// returns true if no error occured, false otherwise.
	#define syclSafeCall(_err_) syclSafeCall2(_err_, __FILE__, __LINE__, FUNCNAME_PORTABLE)

	/// Verifies a cuda call returned "cudaSuccess". Prints error message otherwise.
	/// returns true if no error occured, false otherwise. Calles by cudaSafeCall
	inline bool syclSafeCall2(int err, const char* file, int line, const char* func) {

		//#ifdef CUDA_ERROR_CHECK
		/*
		DPCT1000:1: Error handling if-stmt was detected but could not be rewritten.
		*/
		if (0 != err) {
			char buf[1024];
			/*
			DPCT1001:0: The statement could not be removed.
			*/
			/*
			DPCT1009:2: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
			*/
			sprintf(buf, "SYCL Error (in \"%s\", Line: %d, %s): %d - %s\n", file, line, func, err, "cudaGetErrorString not supported" /*cudaGetErrorString(err)*/);
			printf("%s", buf);
			logging::log_error(buf);
			return false;
		}

		//#endif
		return true;
	}

#ifdef HAVE_CUFFT
	/// Verifies a cuFFT call returned "CUFFT_SUCCESS". Prints error message otherwise.
	/// returns true if no error occured, false otherwise.
    #define cufftSafeCall(_err_) cufftSafeCall2(_err_, __FILE__, __LINE__, FUNCNAME_PORTABLE)

	/// Verifies a cuFFT call returned "CUFFT_SUCCESS". Prints error message otherwise.
	/// returns true if no error occured, false otherwise. Calles by cudaSafeCall
	inline bool cufftSafeCall2(cufftResult err, const char* file, int line, const char* func) {

		//#ifdef CUDA_ERROR_CHECK
		if (CUFFT_SUCCESS != err) {
			char buf[1024];
			sprintf(buf, "CUFFT Error (in \"%s\", Line: %d, %s): %d\n", file, line, func, err);
			printf("%s", buf);
			logging::log_error(buf);
			return false;
		}

		//#endif
		return true;
	}
#endif

	/// Returns the square of x. CUDA constexpr version
	template <typename T>
	constexpr inline T squ(const T& x)
	{
		return x*x;
	}
#else
	#define __host__
	#define __device__
#endif

	template <typename T>
	class LimitProxy
	{
	public:
		inline static T max();
		inline static T min();
	};

	template <>
	class LimitProxy<float>
	{
	public:
		inline static float max() { return FLT_MAX; }
		inline static float min() { return -FLT_MAX; }
	};

	template <>
	class LimitProxy<int16_t>
	{
	public:
		inline static int16_t max() { return 32767; }
		inline static int16_t min() { return -32767; }
	};

	template <>
	class LimitProxy<uint8_t>
	{
	public:
		inline static uint8_t max() { return 255; }
		inline static uint8_t min() { return 0; }
	};

	template <typename ResultType, typename InputType>
	ResultType clampCast(const InputType& x)
	{
		return static_cast<ResultType>(std::min(std::max(x, static_cast<InputType>(LimitProxy<ResultType>::min())), static_cast<InputType>(LimitProxy<ResultType>::max())));
	}

	template <typename ResultType, typename InputType>
	struct clampCaster {
		ResultType operator()(const InputType& a) const
		{
			return clampCast<ResultType>(a);
		}
	};
}

#endif // !__CUDAUTILITY_H__
