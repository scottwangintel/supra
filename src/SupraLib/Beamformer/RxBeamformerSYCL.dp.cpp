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
#include "RxBeamformerSYCL.h"
#include "USImage.h"
#include "USRawData.h"
#include "RxSampleBeamformerDelayAndSum.h"
#include "RxSampleBeamformerDelayAndStdDev.h"
#include "RxSampleBeamformerTestSignal.h"
#include "RxBeamformerCommon.h"
#include "utilities/syclUtility.h"
#include <utilities/utility.h>

//TODO ALL ELEMENT/SCANLINE Y positons are actually Z! Change all variable names accordingly
namespace supra
{
	RxBeamformerSYCL::RxBeamformerSYCL(const RxBeamformerParameters & parameters)
		: m_windowFunction(nullptr)
	{
  
		m_lastSeenDt = 0;
		m_numRxScanlines = parameters.getNumRxScanlines();
		m_rxScanlineLayout = parameters.getRxScanlineLayout();

		m_is3D = (m_rxScanlineLayout.x > 1 && m_rxScanlineLayout.y > 1);
		m_speedOfSoundMMperS = parameters.getSpeedOfSoundMMperS();
		m_rxNumDepths = parameters.getRxNumDepths();

		dpct::device_ext& dev_ct1 = dpct::get_current_device();
  		sycl::queue&	  q_ct1 = dev_ct1.default_queue();
		
		// create and fill new buffers
		m_pRxDepths = std::unique_ptr<Container<LocationType>>(new Container<LocationType>(LocationGpu, &q_ct1, parameters.getRxDepths()));

		m_pRxScanlines = std::unique_ptr<Container<ScanlineRxParameters3D>>(new Container<ScanlineRxParameters3D>(LocationGpu, &q_ct1, parameters.getRxScanlines()));

		m_pRxElementXs = std::unique_ptr<Container<LocationType>>(new Container<LocationType>(LocationGpu, &q_ct1, parameters.getRxElementXs()));
		m_pRxElementYs = std::unique_ptr<Container<LocationType>>(new Container<LocationType>(LocationGpu, &q_ct1, parameters.getRxElementYs()));
	}

	RxBeamformerSYCL::~RxBeamformerSYCL()
	{
	}

	void RxBeamformerSYCL::convertToDtSpace(double dt, double speedOfSoundMMperS, size_t numTransducerElements) const
	{
		if (m_lastSeenDt != dt || m_speedOfSoundMMperS != speedOfSoundMMperS)
		{
			double oldFactor = 1;
			double oldFactorTime = 1;
			if (m_lastSeenDt != 0 && m_speedOfSoundMMperS != 0)
			{
				oldFactor = 1 / (m_speedOfSoundMMperS * m_lastSeenDt);
				oldFactorTime = 1 / m_lastSeenDt;
			}

			double factor = 1 / oldFactor / (speedOfSoundMMperS * dt);
			double factorTime = 1 / oldFactorTime / dt;

			m_pRxScanlines = std::unique_ptr<Container<ScanlineRxParameters3D> >(new Container<ScanlineRxParameters3D>(LocationHost, *m_pRxScanlines));
			for (size_t i = 0; i < m_numRxScanlines; i++)
			{
				ScanlineRxParameters3D p = m_pRxScanlines->get()[i];
				p.position = p.position*factor;
				for (size_t k = 0; k < std::extent<decltype(p.txWeights)>::value; k++)
				{
					p.txParameters[k].initialDelay *= factorTime;
				}
				p.maxElementDistance = p.maxElementDistance*factor;
				m_pRxScanlines->get()[i] = p;
			}
			m_pRxScanlines = std::unique_ptr<Container<ScanlineRxParameters3D> >(new Container<ScanlineRxParameters3D>(LocationGpu, *m_pRxScanlines));

			m_pRxDepths = std::unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, *m_pRxDepths));
			for (size_t i = 0; i < m_rxNumDepths; i++)
			{
				m_pRxDepths->get()[i] = static_cast<LocationType>(m_pRxDepths->get()[i] * factor);
			}
			m_pRxDepths = std::unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, *m_pRxDepths));

			m_pRxElementXs = std::unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, *m_pRxElementXs));
			m_pRxElementYs = std::unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationHost, *m_pRxElementYs));
			for (size_t i = 0; i < numTransducerElements; i++)
			{
				m_pRxElementXs->get()[i] = static_cast<LocationType>(m_pRxElementXs->get()[i] * factor);
				m_pRxElementYs->get()[i] = static_cast<LocationType>(m_pRxElementYs->get()[i] * factor);
			}
			m_pRxElementXs = std::unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, *m_pRxElementXs));
			m_pRxElementYs = std::unique_ptr<Container<LocationType> >(new Container<LocationType>(LocationGpu, *m_pRxElementYs));

			m_lastSeenDt = dt;
			m_speedOfSoundMMperS = speedOfSoundMMperS;
		}
	}

	template <class SampleBeamformer, bool interpolateRFlines, bool interpolateBetweenTransmits, unsigned int maxNumElements, unsigned int maxNumFunctionElements, typename RFType, typename ResultType, typename LocationType>
	
		void rxBeamformingDTSPACE3DKernel(
			uint32_t numTransducerElements,
			vec2T<uint32_t> elementLayout,
			uint32_t numReceivedChannels,
			uint32_t numTimesteps,
			const RFType* __restrict__ RF,
			uint32_t numTxScanlines,
			uint32_t numRxScanlines,
			const ScanlineRxParameters3D* __restrict__ scanlinesDT,
			uint32_t numDs,
			const LocationType* __restrict__ dsDT,
			const LocationType* __restrict__ x_elemsDT,
			const LocationType* __restrict__ z_elemsDT,
			LocationType speedOfSound,
			LocationType dt,
			uint32_t additionalOffset,
			LocationType F,
			const WindowFunctionGpu windowFunction,
			ResultType* __restrict__ s,
			sycl::nd_item<3> item_ct1,
			LocationType *x_elemsDTsh,
			LocationType *z_elemsDTsh,
			WindowFunction::ElementType *functionShared)
	{

		//fetch element positions to shared memory
		for (int threadId = (item_ct1.get_local_id(1) * item_ct1.get_local_range().get(2)) + item_ct1.get_local_id(2);	//@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
			 threadId < maxNumElements && threadId < numTransducerElements;
			 threadId += item_ct1.get_local_range().get(2) * item_ct1.get_local_range().get(1))	 //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
		{
			x_elemsDTsh[threadId] = x_elemsDT[threadId];
			z_elemsDTsh[threadId] = z_elemsDT[threadId];
		}
		for (int threadId = (item_ct1.get_local_id(1) * item_ct1.get_local_range().get(2)) + item_ct1.get_local_id(2);	//@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
			 threadId < maxNumFunctionElements && threadId < windowFunction.numElements();
			 threadId += item_ct1.get_local_range().get(2) * item_ct1.get_local_range().get(1))	 //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
		{
			functionShared[threadId] = windowFunction.getDirect(threadId);
		}
		item_ct1.barrier();	 //@suppress("Function cannot be resolved")

		int r = item_ct1.get_local_range().get(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);			 //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
		int scanlineIdx = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);	 //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")

		if (r < numDs && scanlineIdx < numRxScanlines)
		{
			LocationType d = dsDT[r];
			//TODO should this also depend on the angle?
			LocationType aDT = squ(computeAperture_D(F, d*dt*speedOfSound) / speedOfSound / dt);
			ScanlineRxParameters3D scanline = scanlinesDT[scanlineIdx];

			LocationType scanline_x = scanline.position.x;
			LocationType scanline_z = scanline.position.z;
			LocationType dirX = scanline.direction.x;
			LocationType dirY = scanline.direction.y;
			LocationType dirZ = scanline.direction.z;
			vec2f maxElementDistance = static_cast<vec2f>(scanline.maxElementDistance);
			vec2f		 invMaxElementDistance = vec2f{ 1.0f, 1.0f } / min(vec2f{ sycl::sqrt(aDT), sycl::sqrt(aDT) }, maxElementDistance);

			float sInterp = 0.0f;

			int highestWeightIndex;
			if (!interpolateBetweenTransmits)
			{
				highestWeightIndex = 0;
				float highestWeight = scanline.txWeights[0];
				for (int k = 1; k < std::extent<decltype(scanline.txWeights)>::value; k++)
				{
					if (scanline.txWeights[k] > highestWeight)
					{
						highestWeight = scanline.txWeights[k];
						highestWeightIndex = k;
					}
				}
			}

			// now iterate over all four txScanlines to interpolate beamformed scanlines from those transmits
			for (int k = (interpolateBetweenTransmits ? 0 : highestWeightIndex);
				(interpolateBetweenTransmits && k < std::extent<decltype(scanline.txWeights)>::value) ||
				(!interpolateBetweenTransmits && k == highestWeightIndex);
				k++)
			{
				if (scanline.txWeights[k] > 0.0)
				{
					ScanlineRxParameters3D::TransmitParameters txParams = scanline.txParameters[k];
					uint32_t txScanlineIdx = txParams.txScanlineIdx;
					if (txScanlineIdx >= numTxScanlines)
					{
						//ERROR!
						return;
					}
					float sLocal = 0.0f;
					
					sLocal = SampleBeamformer::template sampleBeamform3D<interpolateRFlines, RFType, float, LocationType>(
						txParams, RF, elementLayout, numReceivedChannels, numTimesteps,
						x_elemsDTsh, z_elemsDTsh, scanline_x, scanline_z, dirX, dirY, dirZ,
						aDT, d, invMaxElementDistance, speedOfSound, dt, additionalOffset, &windowFunction, functionShared);

					if (interpolateBetweenTransmits)
					{
						sInterp += static_cast<float>(scanline.txWeights[k])* sLocal;
					}
					else
					{
						sInterp += sLocal;
					}
				}
			}
			s[scanlineIdx + r * numRxScanlines] = clampCast<ResultType>(sInterp);
		}
	}

	template <class SampleBeamformer, bool interpolateRFlines, bool interpolateBetweenTransmits, typename RFType, typename ResultType, typename LocationType>
	
		void rxBeamformingDTSPACEKernel(
			size_t numTransducerElements,
			size_t numReceivedChannels,
			size_t numTimesteps,
			const RFType* __restrict__ RF,
			size_t numTxScanlines,
			size_t numRxScanlines,
			const ScanlineRxParameters3D* __restrict__ scanlinesDT,
			size_t numDs,
			const LocationType* __restrict__ dsDT,
			const LocationType* __restrict__ x_elemsDT,
			LocationType speedOfSound,
			LocationType dt,
			uint32_t additionalOffset,
			LocationType F,
			const WindowFunctionGpu windowFunction,
			ResultType* __restrict__ s,
			sycl::nd_item<3> item_ct1)
	{
		int r = item_ct1.get_local_range().get(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);			 //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
		int scanlineIdx = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);	 //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
		if (r < numDs && scanlineIdx < numRxScanlines)
		{
			LocationType d = dsDT[r];
			//TODO should this also depend on the angle?
			LocationType aDT = computeAperture_D(F, d*dt*speedOfSound) / speedOfSound / dt;
			ScanlineRxParameters3D scanline = scanlinesDT[scanlineIdx];
			LocationType scanline_x = scanline.position.x;
			LocationType dirX = scanline.direction.x;
			LocationType dirY = scanline.direction.y;
			LocationType dirZ = scanline.direction.z;
			LocationType maxElementDistance = static_cast<LocationType>(scanline.maxElementDistance.x);
			LocationType		   invMaxElementDistance = 1 / sycl::min(aDT, maxElementDistance);

			float sInterp = 0.0f;

			int highestWeightIndex;
			if (!interpolateBetweenTransmits)
			{
				highestWeightIndex = 0;
				float highestWeight = scanline.txWeights[0];
				for (int k = 1; k < std::extent<decltype(scanline.txWeights)>::value; k++)
				{
					if (scanline.txWeights[k] > highestWeight)
					{
						highestWeight = scanline.txWeights[k];
						highestWeightIndex = k;
					}
				}
			}

			// now iterate over all four txScanlines to interpolate beamformed scanlines from those transmits
			for (int k = (interpolateBetweenTransmits ? 0 : highestWeightIndex);
				(interpolateBetweenTransmits && k < std::extent<decltype(scanline.txWeights)>::value) ||
				(!interpolateBetweenTransmits && k == highestWeightIndex);
				k++)
			{
				if (scanline.txWeights[k] > 0.0)
				{
					ScanlineRxParameters3D::TransmitParameters txParams = scanline.txParameters[k];
					uint32_t txScanlineIdx = txParams.txScanlineIdx;
					if (txScanlineIdx >= numTxScanlines)
					{
						//ERROR!
						return;
					}

					float sLocal = 0.0f;
					sLocal = SampleBeamformer::template sampleBeamform2D<interpolateRFlines, RFType, float, LocationType>(
						txParams, RF, numTransducerElements, numReceivedChannels, numTimesteps,
						x_elemsDT, scanline_x, dirX, dirY, dirZ,
						aDT, d, invMaxElementDistance, speedOfSound, dt, additionalOffset, &windowFunction);

					if (interpolateBetweenTransmits)
					{
						sInterp += static_cast<float>(scanline.txWeights[k])* sLocal;
					}
					else
					{
						sInterp += sLocal;
					}
				}
			}
			s[scanlineIdx + r * numRxScanlines] = clampCast<ResultType>(sInterp);
		}
	}


	const int ROW_SIZE = 2;
	template <class SampleBeamformer, bool interpolateRFlines, bool interpolateBetweenTransmits, typename RFType, typename ResultType, typename LocationType>
	void mixed_rxBeamformingDTSPACEKernel(size_t numTransducerElements, size_t numReceivedChannels, size_t numTimesteps, const RFType* __restrict__ RF, size_t numTxScanlines, size_t numRxScanlines,
		const ScanlineRxParameters3D* __restrict__ scanlinesDT, size_t numDs, const LocationType* __restrict__ dsDT, const LocationType* __restrict__ x_elemsDT, LocationType speedOfSound,
		LocationType dt, uint32_t additionalOffset, LocationType F, const WindowFunctionGpu* __restrict__ windowFunction, ResultType* __restrict__ s,   sycl::nd_item<3> item_ct1, float* __restrict__ mdataGpu)
	{
		int r = item_ct1.get_local_range().get(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);			 //@suppress("Symbol is not resolved")
																												 //@suppress("Field cannot be resolved")
		int scanlineIdx = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);	 //@suppress("Symbol is not resolved")
		
																												 //@suppress("Field cannot be resolved")
		const int row_size = ROW_SIZE;
		r = r * row_size;
		if (r < numDs && scanlineIdx < numRxScanlines)
		{
			sycl::vec<LocationType, row_size> d(0.0);
			#pragma unroll
			for (int i = 0; i < row_size; i++) {
				d[i] = dsDT[r + i];
			}
			// TODO should this also depend on the angle?
			sycl::vec<float, row_size> aDT = d / (2*F);
			ScanlineRxParameters3D scanline = scanlinesDT[ scanlineIdx ];
			LocationType scanline_x = scanline.position.x;
			LocationType  dirX = scanline.direction.x;
			LocationType  dirY = scanline.direction.y;
			LocationType  dirZ = scanline.direction.z;
			LocationType  maxElementDistance = static_cast<LocationType>(scanline.maxElementDistance.x);

			sycl::vec<float, row_size> sInterp(0.0);
			
			int highestWeightIndex;
			if (!interpolateBetweenTransmits)
			{
				highestWeightIndex = 0;
				float highestWeight = scanline.txWeights[ 0 ];
				for (int k = 1; k < std::extent<decltype(scanline.txWeights)>::value; k++)
				{
					if (scanline.txWeights[ k ] > highestWeight)
					{
						highestWeight = scanline.txWeights[ k ];
						highestWeightIndex = k;
					}
				}
			}

			// now iterate over all four txScanlines to interpolate beamformed scanlines from those transmits
			for (int k = (interpolateBetweenTransmits ? 0 : highestWeightIndex);
				 (interpolateBetweenTransmits && k < std::extent<decltype(scanline.txWeights)>::value) || (!interpolateBetweenTransmits && k == highestWeightIndex); k++)
			{
				if (scanline.txWeights[ k ] > 0.0)
				{
					ScanlineRxParameters3D::TransmitParameters txParams = scanline.txParameters[ k ];
					uint32_t txScanlineIdx = txParams.txScanlineIdx;
					if (txScanlineIdx >= numTxScanlines)
					{
						// ERROR!
						return;
					}

					#pragma unroll
					for (int i = 0; i < row_size; i++) {
						LocationType invMaxElementDistance = 1.f / sycl::min(aDT[i], maxElementDistance);
						sInterp[i] = SampleBeamformer::template vec_sampleBeamform2D<interpolateRFlines, RFType, float, LocationType>(txParams, RF, numTransducerElements, 
						numReceivedChannels, numTimesteps, x_elemsDT, scanline_x, dirX, dirY, dirZ, aDT[i], d[i], invMaxElementDistance , speedOfSound, dt, additionalOffset, 
						windowFunction, mdataGpu);
					}
					
				}
			}
			
			#pragma unroll
			for (int i = 0; i < row_size; i++) {
				s[scanlineIdx + (r + i) * numRxScanlines] = clampCast<ResultType>(sInterp[i]);
			}
		}
	}



	template <class SampleBeamformer, unsigned int maxWindowFunctionNumel, typename RFType, typename ResultType, typename LocationType>
	void rxBeamformingDTspaceSYCL3D(bool interpolateRFlines, bool interpolateBetweenTransmits, size_t numTransducerElements, vec2s elementLayout, size_t numReceivedChannels, size_t numTimesteps,
									const RFType* RF, size_t numTxScanlines, size_t numRxScanlines, const ScanlineRxParameters3D* scanlines, size_t numZs, const LocationType* zs,
									const LocationType* x_elems, const LocationType* y_elems, LocationType speedOfSound, LocationType dt, uint32_t additionalOffset, LocationType F,
									const WindowFunctionGpu windowFunction, sycl::queue* stream, ResultType* s)
	{
		sycl::range<3> blockSize(1, 256, 1);
		sycl::range<3> gridSize(1, static_cast<unsigned int>((numZs + blockSize[ 1 ] - 1) / blockSize[ 1 ]), static_cast<unsigned int>((numRxScanlines + blockSize[ 2 ] - 1) / blockSize[ 2 ]));

		if (interpolateRFlines)
		{
			if (interpolateBetweenTransmits)
			{
				
						stream->submit([ & ](sycl::handler& cgh) {
							  sycl::accessor<LocationType, 1, sycl::access::mode::read_write, sycl::access::target::local>				  x_elemsDTsh_acc_ct1(sycl::range<1>(1024), cgh);
							  sycl::accessor<LocationType, 1, sycl::access::mode::read_write, sycl::access::target::local>				  z_elemsDTsh_acc_ct1(sycl::range<1>(1024), cgh);
							  sycl::accessor<WindowFunction::ElementType, 1, sycl::access::mode::read_write, sycl::access::target::local> functionShared_acc_ct1(sycl::range<1>(maxWindowFunctionNumel),
																																								 cgh);

							  cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), [ = ](sycl::nd_item<3> item_ct1) {
									rxBeamformingDTSPACE3DKernel<SampleBeamformer, true, true, 1024, maxWindowFunctionNumel>(
										( uint32_t )numTransducerElements, static_cast<vec2T<uint32_t>>(elementLayout), ( uint32_t )numReceivedChannels, ( uint32_t )numTimesteps, RF,
										( uint32_t )numTxScanlines, ( uint32_t )numRxScanlines, scanlines, ( uint32_t )numZs, zs, x_elems, y_elems, speedOfSound, dt, additionalOffset, F,
										windowFunction, s, item_ct1, ( LocationType* )x_elemsDTsh_acc_ct1.get_pointer(), ( LocationType* )z_elemsDTsh_acc_ct1.get_pointer(),
										functionShared_acc_ct1.get_pointer());
							  });
						});
			}
			else {
				
						stream->submit([ & ](sycl::handler& cgh) {
							  sycl::accessor<LocationType, 1, sycl::access::mode::read_write, sycl::access::target::local>				  x_elemsDTsh_acc_ct1(sycl::range<1>(1024), cgh);
							  sycl::accessor<LocationType, 1, sycl::access::mode::read_write, sycl::access::target::local>				  z_elemsDTsh_acc_ct1(sycl::range<1>(1024), cgh);
							  sycl::accessor<WindowFunction::ElementType, 1, sycl::access::mode::read_write, sycl::access::target::local> functionShared_acc_ct1(sycl::range<1>(maxWindowFunctionNumel),
																																								 cgh);

							  cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), [ = ](sycl::nd_item<3> item_ct1) {
									rxBeamformingDTSPACE3DKernel<SampleBeamformer, true, false, 1024, maxWindowFunctionNumel>(
										( uint32_t )numTransducerElements, static_cast<vec2T<uint32_t>>(elementLayout), ( uint32_t )numReceivedChannels, ( uint32_t )numTimesteps, RF,
										( uint32_t )numTxScanlines, ( uint32_t )numRxScanlines, scanlines, ( uint32_t )numZs, zs, x_elems, y_elems, speedOfSound, dt, additionalOffset, F,
										windowFunction, s, item_ct1, ( LocationType* )x_elemsDTsh_acc_ct1.get_pointer(), ( LocationType* )z_elemsDTsh_acc_ct1.get_pointer(),
										functionShared_acc_ct1.get_pointer());
							  });
						});
			}
		}
		else {
			if (interpolateBetweenTransmits)
			{
				
						stream->submit([ & ](sycl::handler& cgh) {
							  sycl::accessor<LocationType, 1, sycl::access::mode::read_write, sycl::access::target::local>				  x_elemsDTsh_acc_ct1(sycl::range<1>(1024), cgh);
							  sycl::accessor<LocationType, 1, sycl::access::mode::read_write, sycl::access::target::local>				  z_elemsDTsh_acc_ct1(sycl::range<1>(1024), cgh);
							  sycl::accessor<WindowFunction::ElementType, 1, sycl::access::mode::read_write, sycl::access::target::local> functionShared_acc_ct1(sycl::range<1>(maxWindowFunctionNumel),
																																								 cgh);

							  cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), [ = ](sycl::nd_item<3> item_ct1) {
									rxBeamformingDTSPACE3DKernel<SampleBeamformer, false, true, 1024, maxWindowFunctionNumel>(
										( uint32_t )numTransducerElements, static_cast<vec2T<uint32_t>>(elementLayout), ( uint32_t )numReceivedChannels, ( uint32_t )numTimesteps, RF,
										( uint32_t )numTxScanlines, ( uint32_t )numRxScanlines, scanlines, ( uint32_t )numZs, zs, x_elems, y_elems, speedOfSound, dt, additionalOffset, F,
										windowFunction, s, item_ct1, ( LocationType* )x_elemsDTsh_acc_ct1.get_pointer(), ( LocationType* )z_elemsDTsh_acc_ct1.get_pointer(),
										functionShared_acc_ct1.get_pointer());
							  });
						});
			}
			else {
				
						stream->submit([ & ](sycl::handler& cgh) {
							  sycl::accessor<LocationType, 1, sycl::access::mode::read_write, sycl::access::target::local>				  x_elemsDTsh_acc_ct1(sycl::range<1>(1024), cgh);
							  sycl::accessor<LocationType, 1, sycl::access::mode::read_write, sycl::access::target::local>				  z_elemsDTsh_acc_ct1(sycl::range<1>(1024), cgh);
							  sycl::accessor<WindowFunction::ElementType, 1, sycl::access::mode::read_write, sycl::access::target::local> functionShared_acc_ct1(sycl::range<1>(maxWindowFunctionNumel),
																																								 cgh);

							  cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), [ = ](sycl::nd_item<3> item_ct1) {
									rxBeamformingDTSPACE3DKernel<SampleBeamformer, false, false, 1024, maxWindowFunctionNumel>(
										( uint32_t )numTransducerElements, static_cast<vec2T<uint32_t>>(elementLayout), ( uint32_t )numReceivedChannels, ( uint32_t )numTimesteps, RF,
										( uint32_t )numTxScanlines, ( uint32_t )numRxScanlines, scanlines, ( uint32_t )numZs, zs, x_elems, y_elems, speedOfSound, dt, additionalOffset, F,
										windowFunction, s, item_ct1, ( LocationType* )x_elemsDTsh_acc_ct1.get_pointer(), ( LocationType* )z_elemsDTsh_acc_ct1.get_pointer(),
										functionShared_acc_ct1.get_pointer());
							  });
						});
			}
		}
		
	}

	template <class SampleBeamformer, typename RFType, typename ResultType, typename LocationType>
	void rxBeamformingDTspaceSYCL(bool interpolateRFlines, bool interpolateBetweenTransmits, size_t numTransducerElements, size_t numReceivedChannels, size_t numTimesteps, const RFType* RF,
								  size_t numTxScanlines, size_t numRxScanlines, const ScanlineRxParameters3D* scanlines, size_t numZs, const LocationType* zs, const LocationType* x_elems,
								  LocationType speedOfSound, LocationType dt, uint32_t additionalOffset, LocationType F, const WindowFunctionGpu windowFunction, sycl::queue* stream, ResultType* s, LocationType *mdataGpu)
	{
		sycl::range<3> blockSize(1, 256, 1);
		sycl::range<3> gridSize(1, static_cast<unsigned int>((numZs + blockSize[ 1 ] - 1) / blockSize[ 1 ] / ROW_SIZE), 
								static_cast<unsigned int>((numRxScanlines + blockSize[ 2 ] - 1) / blockSize[ 2 ]));
		if (interpolateRFlines)
		{
			if (interpolateBetweenTransmits)
			{
				
						stream->submit([ & ](sycl::handler& cgh) {
							  cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), [ = ](sycl::nd_item<3> item_ct1) {
									rxBeamformingDTSPACEKernel<SampleBeamformer, true, true>(numTransducerElements, numReceivedChannels, numTimesteps, RF, numTxScanlines, numRxScanlines, scanlines,
																							 numZs, zs, x_elems, speedOfSound, dt, additionalOffset, F, windowFunction, s, item_ct1);
							  });
						});
			}
			else {

						static long beam_call_count = 0;
				
						sycl::event beam_event = stream->submit([ & ](sycl::handler& cgh) {
							  cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), [ = ](sycl::nd_item<3> item_ct1) {
									/*rxBeamformingDTSPACEKernel<SampleBeamformer, true, false>(numTransducerElements, numReceivedChannels, numTimesteps, RF, numTxScanlines, numRxScanlines, scanlines,
																							  numZs, zs, x_elems, speedOfSound, dt, additionalOffset, F, windowFunction, s, item_ct1);*/
							  
							  		mixed_rxBeamformingDTSPACEKernel<SampleBeamformer, true, false>(numTransducerElements, numReceivedChannels, numTimesteps, RF, numTxScanlines, numRxScanlines,
																								   scanlines, numZs, zs, x_elems, speedOfSound, dt, additionalOffset, F, &windowFunction, s, item_ct1, mdataGpu);
									 
							  
							  });
						});

						beam_event.wait();
						beam_call_count++;
						std::string msg = "Beamforming run " + std::to_string(beam_call_count) + " times: ";
						Report_time(msg, beam_event);
			}
		}
		else {
			if (interpolateBetweenTransmits)
			{
				
						stream->submit([ & ](sycl::handler& cgh) {
							  cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), [ = ](sycl::nd_item<3> item_ct1) {
									rxBeamformingDTSPACEKernel<SampleBeamformer, false, true>(numTransducerElements, numReceivedChannels, numTimesteps, RF, numTxScanlines, numRxScanlines, scanlines,
																							  numZs, zs, x_elems, speedOfSound, dt, additionalOffset, F, windowFunction, s, item_ct1);
							  });
						});
			}
			else {
				
						stream->submit([ & ](sycl::handler& cgh) {
							  cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), [ = ](sycl::nd_item<3> item_ct1) {
									rxBeamformingDTSPACEKernel<SampleBeamformer, false, false>(numTransducerElements, numReceivedChannels, numTimesteps, RF, numTxScanlines, numRxScanlines, scanlines,
																							   numZs, zs, x_elems, speedOfSound, dt, additionalOffset, F, windowFunction, s, item_ct1);
							  });
						});
			}
		}
		
	}

	template <typename ChannelDataType, typename ImageDataType>
	shared_ptr<USImage> RxBeamformerSYCL::performRxBeamforming(
		RxBeamformerSYCL::RxSampleBeamformer sampleBeamformer,
		shared_ptr<const USRawData> rawData,
		double fNumber,
		double speedOfSoundMMperS,
		WindowType windowType,
		WindowFunction::ElementType windowParameter,
		bool interpolateBetweenTransmits,
		int32_t additionalOffset) const
	{
		//Ensure the raw-data are on the gpu
		auto gRawData = rawData->getData<ChannelDataType>();
		if (!gRawData->isGPU() && !gRawData->isBoth())
		{
			gRawData = std::make_shared<Container<ChannelDataType> >(LocationGpu, *gRawData);
		}

		size_t numelOut = m_numRxScanlines*m_rxNumDepths;
		shared_ptr<Container<ImageDataType> > pData = std::make_shared<Container<ImageDataType> >(ContainerLocation::LocationGpu, gRawData->getStream(), numelOut);

		double dt = 1.0 / rawData->getSamplingFrequency();

		if (!m_windowFunction || m_windowFunction->getType() != windowType || m_windowFunction->getParameter() != windowParameter)
		{
			m_windowFunction = std::unique_ptr<WindowFunction>(new WindowFunction(windowType, windowParameter, m_windowFunctionNumEntries));
		}

		auto mdataGpu = (float*) sycl::malloc_device(m_windowFunctionNumEntries * sizeof(windowType), gRawData->getStream()->get_device(), gRawData->getStream()->get_context());
		gRawData->getStream()->submit([&] (sycl::handler &h) {
			h.memcpy(mdataGpu, m_windowFunction->m_data.data(), m_windowFunctionNumEntries * sizeof(WindowType));
		});
		gRawData->getStream()->wait();

		auto beamformingFunction3D = &rxBeamformingDTspaceSYCL3D<RxSampleBeamformerDelayAndSum, m_windowFunctionNumEntries, ChannelDataType, ImageDataType, LocationType>;
		auto beamformingFunction2D = &rxBeamformingDTspaceSYCL<RxSampleBeamformerDelayAndSum, ChannelDataType, ImageDataType, LocationType>;
		


		convertToDtSpace(dt, speedOfSoundMMperS, rawData->getNumElements());
		if (m_is3D)
		{
			beamformingFunction3D(
				true,
				interpolateBetweenTransmits,
				rawData->getNumElements(),
				rawData->getElementLayout(),
				rawData->getNumReceivedChannels(),
				rawData->getNumSamples(),
				gRawData->get(),
				rawData->getNumScanlines(), // numTxScanlines
				m_numRxScanlines,			// numRxScanlines
				m_pRxScanlines->get(),
				m_rxNumDepths, m_pRxDepths->get(),
				m_pRxElementXs->get(),
				m_pRxElementYs->get(),
				static_cast<LocationType>(m_speedOfSoundMMperS),
				static_cast<LocationType>(dt),
				additionalOffset,
				static_cast<LocationType>(fNumber),
				*(m_windowFunction->getGpu()),
				gRawData->getStream(),
				pData->get()
				);
		}
		else {
			beamformingFunction2D(
				true,
				interpolateBetweenTransmits,
				rawData->getNumElements(),
				rawData->getNumReceivedChannels(),
				rawData->getNumSamples(),
				gRawData->get(),
				rawData->getNumScanlines(), // numTxScanlines
				m_numRxScanlines,			// numRxScanlines
				m_pRxScanlines->get(),
				m_rxNumDepths, m_pRxDepths->get(),
				m_pRxElementXs->get(),
				static_cast<LocationType>(m_speedOfSoundMMperS),
				static_cast<LocationType>(dt),
				additionalOffset,
				static_cast<LocationType>(fNumber),
				*(m_windowFunction->getGpu()),
				gRawData->getStream(),
				pData->get(),
				mdataGpu
				);
		}

		if (rawData->getImageProperties() != m_lastSeenImageProperties)
		{
			m_lastSeenImageProperties = rawData->getImageProperties();
			shared_ptr<USImageProperties> newProps = std::make_shared<USImageProperties>(*m_lastSeenImageProperties);
			newProps->setScanlineLayout(m_rxScanlineLayout);
			newProps->setNumSamples(m_rxNumDepths);
			newProps->setImageState(USImageProperties::RF);
			m_editedImageProperties = std::const_pointer_cast<const USImageProperties>(newProps);
		}

		auto retImage = std::make_shared<USImage>(
			vec2s{ m_numRxScanlines, m_rxNumDepths },
			pData,
			m_editedImageProperties,
			rawData->getReceiveTimestamp(),
			rawData->getSyncTimestamp());

		return retImage;
	}

	template
	shared_ptr<USImage> RxBeamformerSYCL::performRxBeamforming<int16_t, int16_t>(
		RxBeamformerSYCL::RxSampleBeamformer sampleBeamformer,
		shared_ptr<const USRawData> rawData,
		double fNumber,
		double speedOfSoundMMperS,
		WindowType windowType,
		WindowFunction::ElementType windowParameter,
		bool interpolateBetweenTransmits,
		int32_t additionalOffset) const;
	template
	shared_ptr<USImage> RxBeamformerSYCL::performRxBeamforming<int16_t, float>(
		RxBeamformerSYCL::RxSampleBeamformer sampleBeamformer,
		shared_ptr<const USRawData> rawData,
		double fNumber,
		double speedOfSoundMMperS,
		WindowType windowType,
		WindowFunction::ElementType windowParameter,
		bool interpolateBetweenTransmits,
		int32_t additionalOffset) const;
	template
	shared_ptr<USImage> RxBeamformerSYCL::performRxBeamforming<float, int16_t>(
		RxBeamformerSYCL::RxSampleBeamformer sampleBeamformer,
		shared_ptr<const USRawData> rawData,
		double fNumber,
		double speedOfSoundMMperS,
		WindowType windowType,
		WindowFunction::ElementType windowParameter,
		bool interpolateBetweenTransmits,
		int32_t additionalOffset) const;
	template
	shared_ptr<USImage> RxBeamformerSYCL::performRxBeamforming<float, float>(
		RxBeamformerSYCL::RxSampleBeamformer sampleBeamformer,
		shared_ptr<const USRawData> rawData,
		double fNumber,
		double speedOfSoundMMperS,
		WindowType windowType,
		WindowFunction::ElementType windowParameter,
		bool interpolateBetweenTransmits,
		int32_t additionalOffset) const;
}