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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include <CL/sycl.hpp>
#include "LogCompressor.h"
#include <utilities/utility.h>

#include <cmath>

using namespace std;

namespace supra
{
	template <typename In, typename Out, typename WorkType>
	
	struct thrustLogcompress
	{
		WorkType _inScale;
		WorkType _scaleOverDenominator;

		// Thrust functor that computes
		// signal = log10(1 + a*signal)./log10(1 + a) 
		// of the downscaled (_inMax) input signal
		thrustLogcompress(double dynamicRange, In inMax, Out outMax, double scale)
			: _inScale(static_cast<WorkType>(dynamicRange / inMax))
			, _scaleOverDenominator(static_cast<WorkType>(scale * outMax / sycl::log10(dynamicRange + 1)))
		{};

		Out operator()(const In& a) const
		{
			WorkType val = sycl::log10(std::abs(static_cast<WorkType>(a))*_inScale + (WorkType)1) * _scaleOverDenominator;
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
		
		auto inImageData_t = inImageData->get();
		auto pComprGpu_t = pComprGpu->get();
		inImageData->getStream()->wait();

		static long log_call_count = 0;
		static std::chrono::duration<double, std::milli> log_total_duration(0);

		sycl::event log_event = inImageData->getStream()->submit([&] (sycl::handler &h) {

			h.parallel_for<>(sycl::range<1>(width * height * depth), [=](sycl::id<1> idx){
				pComprGpu_t[idx] = c(inImageData_t[idx]);
			});
			 
		});

		inImageData->getStream()->wait();
		log_event.wait();
		log_call_count++;
		std::string Log_msg = "Log run " + std::to_string(log_call_count) + " times: ";
		Report_time(Log_msg, log_event);

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