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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "LogCompressor.h"

#include <dpct/dpl_utils.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <cmath>

using namespace std;

namespace supra
{
	template <typename In, typename Out, typename WorkType>
	/*
	DPCT1044:29: thrust::unary_function was removed because std::unary_function has been deprecated in C++11. You may need to remove references to typedefs from thrust::unary_function in the class
	definition.
	*/
	struct thrustLogcompress {
		WorkType _inScale;
		WorkType _scaleOverDenominator;

		// Thrust functor that computes
		// signal = log10(1 + a*signal)./log10(1 + a) 
		// of the downscaled (_inMax) input signal
		thrustLogcompress(double dynamicRange, In inMax, Out outMax, double scale)
			: _inScale(static_cast<WorkType>(dynamicRange / inMax))
			, _scaleOverDenominator(static_cast<WorkType>(scale * outMax / log10(dynamicRange + 1)))
		{};

		Out operator()(const In& a) const
		{
			WorkType val = log10(abs(static_cast<WorkType>(a))*_inScale + (WorkType)1) * _scaleOverDenominator;
			return clampCast<Out>(val);
		}
	};

	template <typename InputType, typename OutputType>
	shared_ptr<Container<OutputType> > LogCompressor::compress(const shared_ptr<const Container<InputType>>& inImageData, vec3s size, double dynamicRange, double scale, double inMax)
	{
		size_t width = size.x;
		size_t height = size.y;
		size_t depth = size.z;

		auto pComprGpu = make_shared<Container<OutputType> >(LocationGpu, inImageData->getStream(), width*height*depth);

		OutputType outMax;
		if (std::is_integral<OutputType>::value)
		{
			outMax = std::numeric_limits<OutputType>::max();
		}
		else if (std::is_floating_point<OutputType>::value)
		{
			outMax = static_cast<OutputType>(255.0);
		}

		thrustLogcompress<InputType, OutputType, WorkType> c(sycl::pow<double>(10, (dynamicRange / 20)), static_cast<InputType>(inMax), outMax, scale);
		std::transform(thrust::cuda::par.on(inImageData->getStream()), inImageData->get(), inImageData->get() + (width * height * depth), pComprGpu->get(), c);
		/*
		DPCT1010:28: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
		*/
		cudaSafeCall(0);

		return pComprGpu;
	}

	template
	shared_ptr<Container<uint8_t> > LogCompressor::compress<int16_t, uint8_t>(const shared_ptr<const Container<int16_t> >& inImageData, vec3s size, double dynamicRange, double scale, double inMax);
	template
	shared_ptr<Container<uint8_t> > LogCompressor::compress<float, uint8_t>(const shared_ptr<const Container<float> >& inImageData, vec3s size, double dynamicRange, double scale, double inMax);
	template
	shared_ptr<Container<uint8_t> > LogCompressor::compress<uint8_t, uint8_t>(const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size, double dynamicRange, double scale, double inMax);
	template
	shared_ptr<Container<float> > LogCompressor::compress<int16_t, float>(const shared_ptr<const Container<int16_t> >& inImageData, vec3s size, double dynamicRange, double scale, double inMax);
	template
	shared_ptr<Container<float> > LogCompressor::compress<float, float>(const shared_ptr<const Container<float> >& inImageData, vec3s size, double dynamicRange, double scale, double inMax);
	template
	shared_ptr<Container<float> > LogCompressor::compress<uint8_t, float>(const shared_ptr<const Container<uint8_t> >& inImageData, vec3s size, double dynamicRange, double scale, double inMax);
}