// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2019, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "HilbertFirEnvelope.h"
#include <utilities/utility.h>
#include <utilities/FirFilterFactory.h>
#include "helper.h"

#include <dpct/dpl_utils.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include <algorithm>

using namespace std;

namespace supra
{
	template <typename InputType, typename OutputType>
	void kernelFilterDemodulation(
		const InputType* __restrict__ signal,
		const HilbertFirEnvelope::WorkType * __restrict__ filter,
		OutputType * __restrict__ out,
		const int numSamples,
		const int numScanlines,
		const int filterLength,
		sycl::nd_item<3> item_ct1) {
		int scanlineIdx = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
		int sampleIdx = item_ct1.get_local_range().get(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);

		if (scanlineIdx < numScanlines && sampleIdx < numSamples)
		{
			HilbertFirEnvelope::WorkType accumulator = 0;
			
			int startPoint = sampleIdx - filterLength / 2;
			int endPoint = sampleIdx + filterLength / 2;
			int currentFilterElement = 0;
			for (int currentSample = startPoint;
				currentSample <= endPoint;
				currentSample++, currentFilterElement++)
			{
				if (currentSample >= 0 && currentSample < numSamples)
				{
					HilbertFirEnvelope::WorkType sample = static_cast<HilbertFirEnvelope::WorkType>(signal[scanlineIdx + currentSample*numScanlines]);
					HilbertFirEnvelope::WorkType filterElement = filter[currentFilterElement];
					accumulator += sample*filterElement;
				}
			}

			HilbertFirEnvelope::WorkType signalValue = static_cast<HilbertFirEnvelope::WorkType>(signal[scanlineIdx + sampleIdx*numScanlines]);
			out[ scanlineIdx + sampleIdx * numScanlines ] = sycl::sqrt(squ(signalValue) + squ(accumulator));
		}

	}

	const int H_VEC_SIZE = 4;
	template <typename InputType, typename OutputType>
	void vec_kernelFilterDemodulation(
		const InputType* __restrict__ signal,
		const HilbertFirEnvelope::WorkType * __restrict__ filter,
		OutputType * __restrict__ out,
		const int numSamples,
		const int numScanlines,
		const int filterLength,
		sycl::nd_item<3> item_ct1) {

		int scanlineIdx = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
		int sampleIdx = item_ct1.get_local_range().get(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);

		scanlineIdx *= H_VEC_SIZE;
		if (scanlineIdx < numScanlines && sampleIdx < numSamples)
		{
			sycl::vec<HilbertFirEnvelope::WorkType, H_VEC_SIZE> accumulator(0.0);
				
			int startPoint = sampleIdx - filterLength / 2;
			int endPoint = sampleIdx + filterLength / 2;
			int currentFilterElement = 0;

			for (int currentSample = startPoint;
				currentSample <= endPoint;
				currentSample ++, currentFilterElement++)
			{
				if (currentSample >= 0 && currentSample < numSamples)
				{
					sycl::vec<HilbertFirEnvelope::WorkType, H_VEC_SIZE> vec_sample(0.0);
					#pragma unroll
					for (int c = 0; c < H_VEC_SIZE; c++) {
						vec_sample[c] = static_cast<HilbertFirEnvelope::WorkType>(signal[scanlineIdx + c +  currentSample * numScanlines]) 
								* filter[currentFilterElement];
					}
					accumulator += vec_sample;
				}
			}
			#pragma unroll
			for (int c = 0; c < H_VEC_SIZE; c++) {
				HilbertFirEnvelope::WorkType signalValue = static_cast<HilbertFirEnvelope::WorkType>(signal[scanlineIdx + c + sampleIdx*numScanlines]);
				out[ scanlineIdx + c + sampleIdx * numScanlines ] = sycl::sqrt(squ(signalValue) + squ(accumulator[c]));
			}
		}

	}


	HilbertFirEnvelope::HilbertFirEnvelope(size_t filterLength)
		: m_filterLength(filterLength)
		, m_hilbertFilter(nullptr)
	{
		prepareFilter();
	}

	HilbertFirEnvelope::~HilbertFirEnvelope()
	{
	}

	void HilbertFirEnvelope::prepareFilter()
	{
		m_hilbertFilter = FirFilterFactory::createFilter<float>(
			m_filterLength,
			FirFilterFactory::FilterTypeHilbertTransformer,
			FirFilterFactory::FilterWindowHamming);
		m_hilbertFilter = make_shared<Container<float> >(LocationGpu, *m_hilbertFilter);
	}

	template<typename InputType, typename OutputType>
	shared_ptr<Container<OutputType> > HilbertFirEnvelope::demodulate(
		const shared_ptr<const Container<InputType>>& inImageData,
		int numScanlines, int numSamples)
	{
		auto pEnv = make_shared<Container<OutputType> >(LocationGpu, inImageData->getStream(), numScanlines*numSamples);
		sycl::range<3> blockSizeFilter(1, 8, 16);
		sycl::range<3> gridSizeFilter(1, static_cast<unsigned int>((numSamples + blockSizeFilter[ 1 ] - 1) / blockSizeFilter[ 1 ]),
									  static_cast<unsigned int>((numScanlines + blockSizeFilter[ 2 ] - 1) / blockSizeFilter[ 2 ] / H_VEC_SIZE));

				static long hilbert_call_count = 0;

				sycl::event hilbert_event = inImageData->getStream()->submit([ & ](sycl::handler& cgh) {
						auto inImageData_get_ct0 = inImageData->get();
						auto m_hilbertFilter_get_ct1 = m_hilbertFilter->get();
						auto pEnv_get_ct2 = pEnv->get();
						auto m_filterLength_ct5 = ( int )m_filterLength;

						cgh.parallel_for(sycl::nd_range<3>(gridSizeFilter * blockSizeFilter, blockSizeFilter), [ = ](sycl::nd_item<3> item_ct1) {
								vec_kernelFilterDemodulation(inImageData_get_ct0, m_hilbertFilter_get_ct1, pEnv_get_ct2, numSamples, numScanlines, m_filterLength_ct5, item_ct1);
						});
				});

				hilbert_event.wait();
				hilbert_call_count++;
				std::string msg = "Hilbert run " + std::to_string(hilbert_call_count) + " times: ";
				Report_time(msg, hilbert_event);
		

		return pEnv;
	}

	template 
	shared_ptr<Container<int16_t> > HilbertFirEnvelope::demodulate<int16_t, int16_t>(
		const shared_ptr<const Container<int16_t> >& inImageData,
		int numScanlines, int numSamples);
	template
		shared_ptr<Container<int16_t> > HilbertFirEnvelope::demodulate<float, int16_t>(
			const shared_ptr<const Container<float> >& inImageData,
			int numScanlines, int numSamples);
	template
		shared_ptr<Container<float> > HilbertFirEnvelope::demodulate<int16_t, float>(
			const shared_ptr<const Container<int16_t> >& inImageData,
			int numScanlines, int numSamples);
	template
		shared_ptr<Container<float> > HilbertFirEnvelope::demodulate<float, float>(
			const shared_ptr<const Container<float> >& inImageData,
			int numScanlines, int numSamples);
}