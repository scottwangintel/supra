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

#ifndef __RXBEAMFORMERCOMMON_H__
#define __RXBEAMFORMERCOMMON_H__

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "WindowFunction.h"

//TODO ALL ELEMENT/SCANLINE Y positons are actually Z! Change all variable names accordingly
namespace supra
{

	template <typename T>
	inline T computeAperture_D(T F, T z)
	{
		return z / (2 * F);
	}

	template <typename T>
	inline T computeDelayDTSPACE_D(T dirX, T dirY, T dirZ, T x_element, T x, T z)
	{
		return sycl::sqrt((x_element - (x + dirX*z))*
			(x_element - (x + dirX*z)) +
			(dirY*z)*(dirY*z)) + z;
	}

	const int Vec_SIZE = 8;
	sycl::vec<float, Vec_SIZE> vec_computeDelayDTSPACE_D(float dirX, float dirY, float dirZ, sycl::vec<float, Vec_SIZE> x_element, float x, float z)
	{
		return sycl::sqrt(((x_element - (x + dirX * z)) * (x_element - (x + dirX * z)) + (dirY * z) * (dirY * z))) + z;
	}

	template <typename T>
	inline T computeDelayDTSPACE3D_D(T dirX, T dirY, T dirZ, T x_element, T z_element, T x, T z, T d)
	{
		return sycl::sqrt(
			squ(x_element - (x + dirX*d)) +
			squ(z_element - (z + dirZ*d)) +
			squ(dirY*d)) + d;
	}

	// distance has to be normalized to [-1, 1] (inclusive)
	inline WindowFunctionGpu::ElementType
		computeWindow3D(const WindowFunctionGpu& windowFunction, const vec2f& distance)
	{
		return sycl::sqrt(windowFunction.get(distance.x) * windowFunction.get(distance.y));
	}


	// distance has to be normalized to [-1, 1] (inclusive)
	inline WindowFunctionGpu::ElementType
		computeWindow3DShared(const WindowFunctionGpu& windowFunction, const WindowFunctionGpu::ElementType * __restrict__ sharedData, const vec2f& distance)
	{
		return sycl::sqrt(windowFunction.getShared(sharedData, distance.x) * windowFunction.getShared(sharedData, distance.y));
	}
}

#endif //!__RXBEAMFORMERCOMMON_H__