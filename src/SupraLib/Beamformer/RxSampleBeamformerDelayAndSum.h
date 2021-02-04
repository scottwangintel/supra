// ================================================================================================
// 
// If not explicitly stated: Copyright (C) 2017, all rights reserved,
//      Rüdiger Göbl 
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// ================================================================================================

#ifndef __RXSAMPLEBEAMFORMERDELAYANDSUM_H__
#define __RXSAMPLEBEAMFORMERDELAYANDSUM_H__

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "USImageProperties.h"
#include "WindowFunction.h"
#include "RxBeamformerCommon.h"
#include "helper.h"

//TODO ALL ELEMENT/SCANLINE Y positons are actually Z! Change all variable names accordingly
namespace supra
{
	class RxSampleBeamformerDelayAndSum
	{
	public:
		template <bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
		static ResultType sampleBeamform3D(
			ScanlineRxParameters3D::TransmitParameters txParams,
			const RFType* RF,
			vec2T<uint32_t> elementLayout,
			uint32_t numReceivedChannels,
			uint32_t numTimesteps,
			const LocationType* x_elemsDTsh,
			const LocationType* z_elemsDTsh,
			LocationType scanline_x,
			LocationType scanline_z,
			LocationType dirX,
			LocationType dirY,
			LocationType dirZ,
			LocationType aDT,
			LocationType depth,
			vec2f invMaxElementDistance,
			LocationType speedOfSound,
			LocationType dt,
			int32_t additionalOffset,
			const WindowFunctionGpu* windowFunction,
			const WindowFunction::ElementType* functionShared
		)
		{
			float sample = 0.0f;
			float weightAcum = 0.0f;
			int numAdds = 0;
			LocationType initialDelay = txParams.initialDelay;
			uint32_t txScanlineIdx = txParams.txScanlineIdx;
			for (uint32_t elemIdxX = txParams.firstActiveElementIndex.x; elemIdxX < txParams.lastActiveElementIndex.x; elemIdxX++)
			{
				for (uint32_t elemIdxY = txParams.firstActiveElementIndex.y; elemIdxY < txParams.lastActiveElementIndex.y; elemIdxY++)
				{
					uint32_t elemIdx = elemIdxX + elemIdxY*elementLayout.x;
					uint32_t  channelIdx = elemIdx % numReceivedChannels;
					LocationType x_elem = x_elemsDTsh[elemIdx];
					LocationType z_elem = z_elemsDTsh[elemIdx];

					if ((squ(x_elem - scanline_x) + squ(z_elem - scanline_z)) <= aDT)
					{
						vec2f elementScanlineDistance = { x_elem - scanline_x, z_elem - scanline_z };
						float weight = computeWindow3DShared(*windowFunction, functionShared, elementScanlineDistance * invMaxElementDistance);
						weightAcum += weight;
						numAdds++;
						if (interpolateRFlines)
						{
							LocationType delayf = initialDelay +
								computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elem, z_elem, scanline_x, scanline_z, depth) + additionalOffset;
							uint32_t delay = static_cast<uint32_t>(sycl::floor(delayf));
							delayf -= delay;
							if (delay < (numTimesteps - 1))
							{
								sample +=
									weight * ((1.0f - delayf) * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps] +
										delayf  * RF[(delay + 1) + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps]);
							}
							else if (delay < numTimesteps && delayf == 0.0)
							{
								sample += weight * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
							}
						}
						else
						{
							uint32_t delay = static_cast<uint32_t>(sycl::round(
								initialDelay + computeDelayDTSPACE3D_D(dirX, dirY, dirZ, x_elem, z_elem, scanline_x, scanline_z, depth)) + additionalOffset);
							if (delay < numTimesteps)
							{
								sample += weight * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
							}
						}
					}
				}
			}
			if (numAdds > 0)
			{
				return sample / weightAcum * numAdds;
			}
			else
			{
				return 0;
			}
		}

		template <bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
		static ResultType sampleBeamform2D(
			ScanlineRxParameters3D::TransmitParameters txParams,
			const RFType* RF,
			uint32_t numTransducerElements,
			uint32_t numReceivedChannels,
			uint32_t numTimesteps,
			const LocationType* x_elemsDT,
			LocationType scanline_x,
			LocationType dirX,
			LocationType dirY,
			LocationType dirZ,
			LocationType aDT,
			LocationType depth,
			LocationType invMaxElementDistance,
			LocationType speedOfSound,
			LocationType dt,
			int32_t additionalOffset,
			const WindowFunctionGpu* windowFunction
		)
		{
			float sample = 0.0f;
			float weightAcum = 0.0f;
			int numAdds = 0;
			LocationType initialDelay = txParams.initialDelay;
			uint32_t txScanlineIdx = txParams.txScanlineIdx;

			for (int32_t elemIdxX = txParams.firstActiveElementIndex.x; elemIdxX < txParams.lastActiveElementIndex.x; elemIdxX++)
			{
				int32_t  channelIdx = elemIdxX % numReceivedChannels;
				LocationType x_elem = x_elemsDT[elemIdxX];
				if (abs(x_elem - scanline_x) <= aDT)
				{
					float weight = windowFunction->get((x_elem - scanline_x) * invMaxElementDistance);
					weightAcum += weight;
					numAdds++;
					if (interpolateRFlines)
					{
						LocationType delayf = initialDelay +
							computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, depth) + additionalOffset;
						int32_t delay = static_cast<int32_t>(sycl::floor(delayf));
						delayf -= delay;
						if (delay < (numTimesteps - 1))
						{
							sample +=
								weight * ((1.0f - delayf) * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps] +
									delayf  * RF[(delay + 1) + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps]);
						}
						else if (delay < numTimesteps && delayf == 0.0)
						{
							sample += weight * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
						}
					}
					else
					{
						int32_t delay = static_cast<int32_t>(sycl::round(
							initialDelay + computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, depth)) + additionalOffset);
						if (delay < numTimesteps)
						{
							sample += weight * RF[delay + channelIdx*numTimesteps + txScanlineIdx*numReceivedChannels*numTimesteps];
						}
					}
				}
			}
			if (numAdds > 0)
			{
				return sample / weightAcum * numAdds;
			}
			else
			{
				return 0;
			}
		}

		template <bool interpolateRFlines, typename RFType, typename ResultType, typename LocationType>
		static ResultType vec_sampleBeamform2D(
			ScanlineRxParameters3D::TransmitParameters txParams, 
			const RFType* RF, 
			uint32_t numTransducerElements, 
			uint32_t numReceivedChannels,
			uint32_t numTimesteps, 
			const LocationType* x_elemsDT, 
			LocationType scanline_x, 
			LocationType dirX, 
			LocationType dirY, 
			LocationType dirZ, 
			LocationType aDT,
			LocationType depth, 
			LocationType invMaxElementDistance, 
			LocationType speedOfSound, 
			LocationType dt, 
			int32_t additionalOffset,
			const WindowFunctionGpu* __restrict__ windowFunction,
			const float* mdataGpu)
		{
			const int VEC_SIZE = Vec_SIZE;
			float sampleAcum = 0.0f;	
			float weightAcum = 0.0f;
			int numAdds = 0;
			LocationType initialDelay = txParams.initialDelay;
			uint32_t txScanlineIdx = txParams.txScanlineIdx;

			for (int32_t elemIdxX = txParams.firstActiveElementIndex.x; elemIdxX < txParams.lastActiveElementIndex.x; elemIdxX += VEC_SIZE)
			{
				sycl::vec<int, VEC_SIZE> channelIdx;
				sycl::vec<LocationType, VEC_SIZE> x_elem;

				#pragma unroll
				for (int i = 0; i < VEC_SIZE; i +=2) {
					channelIdx[i] = (elemIdxX + i) % numReceivedChannels;
					channelIdx[i+1] = (elemIdxX + i + 1) % numReceivedChannels;
					x_elem[i] = x_elemsDT[elemIdxX + i];
					x_elem[i + 1] = x_elemsDT[elemIdxX + i + 1];
				}
				sycl::vec<float, VEC_SIZE> sample;
				sycl::vec<int, VEC_SIZE> mask = (sycl::fabs(x_elem - scanline_x) <= aDT);
				/*sycl spec1.2.1 mentioned: true return  -1, false return 0*/
				mask *= -1;
				numAdds += utils<int, VEC_SIZE>::add_vec(mask);

				sycl::vec<float, VEC_SIZE> relativeIndex = (x_elem - scanline_x) * invMaxElementDistance;
				sycl::vec<float, VEC_SIZE> relativeIndexClamped = sycl::min(sycl::max(relativeIndex, -1.0f), 1.0f);	
				sycl::vec<float, VEC_SIZE> absoluteIndex = windowFunction->m_scale * (relativeIndexClamped + 1.0f);	
				sycl::vec<int, VEC_SIZE> absoluteIndex_int = absoluteIndex.convert<int, sycl::rounding_mode::automatic>();
				sycl::vec<float, VEC_SIZE> weight;

				#pragma unroll
				for (int i = 0; i < VEC_SIZE; i += 2 ) {
					weight[i] = mdataGpu[absoluteIndex_int[i]];
					weight[i + 1] = mdataGpu[absoluteIndex_int[i + 1]];
				}

				weight *= mask.convert<float, sycl::rounding_mode::automatic>();
				weightAcum += utils<float, VEC_SIZE>::add_vec(weight);

				sycl::vec<LocationType, VEC_SIZE> delayf = initialDelay +
					vec_computeDelayDTSPACE_D(dirX, dirY, dirZ, x_elem, scanline_x, depth) + additionalOffset;
				sycl::vec<float, VEC_SIZE> delay = sycl::floor(delayf);
				sycl::vec<int, VEC_SIZE> delay_index = delay.convert<int, sycl::rounding_mode::automatic>() + channelIdx*numTimesteps + 
					txScanlineIdx*numReceivedChannels*numTimesteps;
				
				delayf -= delay;

				sycl::vec<float, VEC_SIZE> RF_data;
				sycl::vec<float, VEC_SIZE> RF_data_one;
				
				#pragma unroll
				for (int j = 0; j < VEC_SIZE; j += 2) {
					RF_data[j] = (float)RF[delay_index[j]];
					RF_data[j + 1] = (float)RF[delay_index[j + 1]];
					RF_data_one[j] = (float)RF[delay_index[j]+1];
					RF_data_one[j + 1] = (float)RF[delay_index[j + 1]+1];
				}

				sycl::vec<int, VEC_SIZE> mask1 = (delay < (numTimesteps - 1));
				mask1 *= -1;
				sample = weight * ((1.0f - delayf) * RF_data + 
					delayf  * RF_data_one) * mask1.convert<float, sycl::rounding_mode::automatic>();
				sampleAcum += utils<float, VEC_SIZE>::add_vec(sample);
				
				sycl::vec<int, VEC_SIZE> mask2 = (delay < numTimesteps && delayf == 0.0);
				mask2 *= -1;
				sample = weight * RF_data * mask2.convert<float, sycl::rounding_mode::automatic>();
				sampleAcum += utils<float, VEC_SIZE>::add_vec(sample);
				
			}

			if (numAdds > 0)
			{
				return sampleAcum / weightAcum * numAdds;
			}
			else
			{
				return 0;
			}
		}

	};
}

#endif //!__RXSAMPLEBEAMFORMERDELAYANDSUM_H__
