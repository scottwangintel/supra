// ================================================================================================
// 
// Copyright (C) 2016, Rüdiger Göbl - all rights reserved
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
//          Rüdiger Göbl
//          Email r.goebl@tum.de
//          Chair for Computer Aided Medical Procedures
//          Technische Universität München
//          Boltzmannstr. 3, 85748 Garching b. München, Germany
// 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License, version 2.1, as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this program.  If not, see
// <http://www.gnu.org/licenses/>.
//
// ================================================================================================

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ScanConverter.h"
#include <cassert>
#include <utilities/cudaUtility.h>
#include <utilities/Logging.h>

#include <dpct/dpl_utils.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <cmath>

using namespace std;

namespace supra
{
	class ScanConverterInternals {
	public:
		typedef ScanConverter::IndexType IndexType;
		typedef ScanConverter::WeightType WeightType;

		static constexpr double m_tetrahedronTestDistanceThreshold = 1e-9;
		static constexpr int m_mappingMaxIterations = ScanConverter::m_mappingMaxIterations;
		static constexpr double m_mappingDistanceThreshold = ScanConverter::m_mappingDistanceThreshold;

		template <typename Tf, typename Ti>
		static void computeParametersVoxel3D(
			const Tf &sampleDistance,
			const vec2T<Ti> &scanlineLayout,
			const int &scanlineIdxX,
			const int &scanlineIdxY,
			const vec3T<Tf> &s1,
			const vec3T<Tf> &e1,
			const vec3T<Tf> &s2,
			const vec3T<Tf> &e2,
			const vec3T<Tf> &s3,
			const vec3T<Tf> &e3,
			const vec3T<Tf> &s4,
			const vec3T<Tf> &e4,
			const vec3T<Tf> &scanline1Pos,
			const vec3T<Tf> &scanline1Dir,
			const vec3T<Tf> &scanline2Pos,
			const vec3T<Tf> &scanline2Dir,
			const vec3T<Tf> &scanline3Pos,
			const vec3T<Tf> &scanline3Dir,
			const vec3T<Tf> &scanline4Pos,
			const vec3T<Tf> &scanline4Dir,
			const Tf &startDepth,
			const Tf &endDepth,
			const vec3T<Ti> &imageSize,
			const vec3T<Ti> &voxel,
			const vec3T<Tf> &voxelPos,
			uint8_t* __restrict__ maskBuf,
			uint32_t* __restrict__ sampleIdxBuf,
			float* __restrict__ weightXBuf,
			float* __restrict__ weightYBuf,
			float* __restrict__ weightZBuf
		)
		{
			if (pointInsideTetrahedron(s1, s2, s3, e1, voxelPos) ||
				pointInsideTetrahedron(s2, s4, s3, e4, voxelPos) ||
				pointInsideTetrahedron(s2, e1, e2, e4, voxelPos) ||
				pointInsideTetrahedron(s3, e3, e1, e4, voxelPos) ||
				pointInsideTetrahedron(s2, s3, e1, e4, voxelPos))
			{

				thrust::pair<vec3T<Tf>, bool> params = mapToParameters3D<Tf, Ti>(
					scanline1Pos,
					scanline2Pos,
					scanline3Pos,
					scanline4Pos,
					scanline1Dir,
					scanline2Dir,
					scanline3Dir,
					scanline4Dir,
					startDepth, endDepth, voxelPos);

				if (params.second)
				{
					size_t voxelIndex = voxel.x + voxel.y*imageSize.x + voxel.z*imageSize.x*imageSize.y;
					maskBuf[voxelIndex] = 1;

					Tf t1 = params.first.x;
					Tf t2 = params.first.y;
					Tf d = params.first.z + 0;

					IndexType  sampleIdxScanline = static_cast<IndexType>(sycl::floor(d / sampleDistance));
					WeightType weightY = static_cast<WeightType>(d / sampleDistance - sampleIdxScanline);
					WeightType weightX = static_cast<WeightType>(t1);
					WeightType weightZ = static_cast<WeightType>(t2);

					IndexType sampleIdx = static_cast<IndexType>(sampleIdxScanline*scanlineLayout.x*scanlineLayout.y +
						scanlineIdxX + scanlineIdxY*scanlineLayout.x);

					sampleIdxBuf[voxelIndex] = sampleIdx;
					weightXBuf[voxelIndex] = weightX;
					weightYBuf[voxelIndex] = weightY;
					weightZBuf[voxelIndex] = weightZ;
				}
			}
		}

		/**
		 * Tests whether point p lies within the tetrahedron defined by a, b, c, d.
		 *
		 * For the test, the barycentric coordinates of p are computed and checked for equal sign.
		 */
		template <typename Tf>
		static bool pointInsideTetrahedron(const vec3T<Tf> & a, const vec3T<Tf> & b, const vec3T<Tf> & c, const vec3T<Tf> & d, const vec3T<Tf> & p)
		{
			Tf w0 = barycentricCoordinate3D(a, b, c, d);

			Tf w1 = barycentricCoordinate3D(p, b, c, d);
			Tf w2 = barycentricCoordinate3D(a, p, c, d);
			Tf w3 = barycentricCoordinate3D(a, b, p, d);
			Tf w4 = barycentricCoordinate3D(a, b, c, p);

			return w0 > 0 &&
				w1 >= -m_tetrahedronTestDistanceThreshold &&
				w2 >= -m_tetrahedronTestDistanceThreshold &&
				w3 >= -m_tetrahedronTestDistanceThreshold &&
				w4 >= -m_tetrahedronTestDistanceThreshold;
		}

		template <typename Tf>
		static Tf barycentricCoordinate3D(const vec3T<Tf> & a, const vec3T<Tf> & b, const vec3T<Tf> & c, const vec3T<Tf> & p)
		{
			//computes the determinant of 
			//[a_x, a_y, a_z, 1]
			//[b_x, b_y, b_z, 1]
			//[c_x, c_y, c_z, 1]
			//[p_x, p_y, p_z, 1]

			// reducing 12 multiplications per compute
			const Tf axby = a.x*b.y;
			const Tf cypz = c.y*p.z;
			const Tf axbz = a.x*b.z;
			const Tf czpy = c.z*p.y;
			const Tf aybx = a.y*b.x;
			const Tf cxpz = c.x*p.z;
			const Tf aybz = a.y*b.z;
			const Tf czpx = c.z*p.x;
			const Tf azbx = a.z*b.x;
			const Tf cxpy = c.x*p.y;
			const Tf azby = a.z*b.y;
			const Tf cypx = c.y*p.x;

			return 
				(axby-aybx)*(c.z-p.z) + (aybz-azby)*(c.x-p.x) +
				(azbx-axbz)*(c.y-p.y) + (cypz-czpy)*(a.x-b.x) -
				(cxpz-czpx)*(a.y-b.y) + (cxpy-cypx)*(a.z-b.z);
			// reducing 18 multiplications with the updated return statement per compute 			
		}

		template <typename Tf>
		static vec3T<Tf> pointPlaneConnection(const vec3T<Tf> & a, const vec3T<Tf> & na, const vec3T<Tf> & x)
		{
			return dot(na, (x - a))*na;
		}

		template <typename Tf, typename Ti>
		static thrust::pair<vec3T<Tf>, bool> mapToParameters3D(
			const vec3T<Tf> & a,
			const vec3T<Tf> & ax,
			const vec3T<Tf> & ay,
			const vec3T<Tf> & axy,
			const vec3T<Tf> & da,
			const vec3T<Tf> & dax,
			const vec3T<Tf> & day,
			const vec3T<Tf> & daxy,
			Tf startDepth,
			Tf endDepth,
			const vec3T<Tf> & x)
		{
			vec3T<Tf> normalXLow = normalize(cross(da, (ay + day) - a));
			vec3T<Tf> normalYLow = normalize(cross((ax + dax) - a, da));
			vec3T<Tf> normalXHigh = normalize(cross(dax, (axy + daxy) - ax));
			vec3T<Tf> normalYHigh = normalize(cross((axy + daxy) - ay, day));

			//find t via binary search
			vec2T<Tf> lowT = { 0, 0 };
			vec2T<Tf> highT = { 1, 1 };
			vec3T<Tf> lowConnX = pointPlaneConnection(a, normalXLow, x);
			vec3T<Tf> highConnX = pointPlaneConnection(ax, normalXHigh, x);
			vec3T<Tf> lowConnY = pointPlaneConnection(a, normalYLow, x);
			vec3T<Tf> highConnY = pointPlaneConnection(ay, normalYHigh, x);
			vec2T<Tf> lowDist = { norm(lowConnX), norm(lowConnY) };
			vec2T<Tf> highDist = { norm(highConnX), norm(highConnY) };

			if (dot(lowConnX, highConnX) > 0 || dot(lowConnY, highConnY) > 0)
			{
				return thrust::pair<vec3T<Tf>, bool>(vec3T<Tf>{ 0, 0, 0 }, false);
			}

			vec2T<Tf> dist = { 1e10, 1e10 };
			vec2T<Tf> t = (highT - lowT) / 2 + lowT;
			vec3T<Tf> planeBaseX1;
			vec3T<Tf> planeBaseY1;
			vec3T<Tf> planeBaseX2;
			vec3T<Tf> planeBaseY2;
			for (int numIter = 0; numIter < m_mappingMaxIterations &&
				(dist.x > m_mappingDistanceThreshold || dist.y > m_mappingDistanceThreshold); numIter++)
			{
				t = (1 - highDist / (highDist + lowDist))*highT + (1 - lowDist / (highDist + lowDist))*lowT;

				planeBaseX1 = (1 - t.x)*a + t.x*ax;
				planeBaseX2 = (1 - t.x)*ay + t.x*axy;
				planeBaseY1 = (1 - t.y)*a + t.y*ay;
				planeBaseY2 = (1 - t.y)*ax + t.y*axy;
				vec3T<Tf> dir = slerp3(slerp3(da, dax, t.x), slerp3(day, daxy, t.x), t.y);
				vec3T<Tf> normal_x = normalize(cross(dir, planeBaseX2 - planeBaseX1));
				vec3T<Tf> normal_y = normalize(cross(planeBaseY2 - planeBaseY1, dir));

				vec3T<Tf> connX = pointPlaneConnection(planeBaseX1, normal_x, x);
				vec3T<Tf> connY = pointPlaneConnection(planeBaseY1, normal_y, x);

				dist.x = norm(connX);
				dist.y = norm(connY);

				if (dot(highConnX, connX) > M_EPS)
				{
					highT.x = t.x;
					highConnX = connX;
					highDist.x = dist.x;
				}
				else if (dot(lowConnX, connX) > M_EPS)
				{
					lowT.x = t.x;
					lowConnX = connX;
					lowDist.x = dist.x;
				}

				if (dot(highConnY, connY) > M_EPS)
				{
					highT.y = t.y;
					highConnY = connY;
					highDist.y = dist.y;
				}
				else if (dot(lowConnY, connY) > M_EPS)
				{
					lowT.y = t.y;
					lowConnY = connY;
					lowDist.y = dist.y;
				}
			}

			vec3T<Tf> lineBase = (1 - t.y)*planeBaseX1 + t.y*planeBaseX2;
			Tf d = norm(x - lineBase);

			return thrust::pair<vec3T<Tf>, bool>(vec3T<Tf>{ t.x, t.y, d }, true);
		}
	};

	template <typename Tf, typename Ti>
	void
		
		computeParameterBB3D(
			const Tf sampleDistance,
			const vec2T<Ti> scanlineLayout,
			const int scanlineIdxX,
			const int scanlineIdxY,
			const vec3T<Tf> s1,
			const vec3T<Tf> e1,
			const vec3T<Tf> s2,
			const vec3T<Tf> e2,
			const vec3T<Tf> s3,
			const vec3T<Tf> e3,
			const vec3T<Tf> s4,
			const vec3T<Tf> e4,
			const vec3T<Tf> scanline1Pos,
			const vec3T<Tf> scanline1Dir,
			const vec3T<Tf> scanline2Pos,
			const vec3T<Tf> scanline2Dir,
			const vec3T<Tf> scanline3Pos,
			const vec3T<Tf> scanline3Dir,
			const vec3T<Tf> scanline4Pos,
			const vec3T<Tf> scanline4Dir,
			const Tf startDepth,
			const Tf endDepth,
			const vec3T<Ti> imageSize,
			const vec3T<Tf> bbMin,
			const vec3T<Ti> tetMinVoxel,
			const vec3T<Ti> tetMaxVoxel,
			const Tf resolution,
			uint8_t* __restrict__ maskBuf,
			uint32_t* __restrict__ sampleIdxBuf,
			float* __restrict__ weightXBuf,
			float* __restrict__ weightYBuf,
			float* __restrict__ weightZBuf
		,
			sycl::nd_item<3> item_ct1)
	{
		vec3T<Ti> voxel = vec3T<Ti>{ static_cast<Ti>(item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)),
									 static_cast<Ti>(item_ct1.get_local_range().get(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1)),
									 static_cast<Ti>(item_ct1.get_local_range().get(0) * item_ct1.get_group(0)
													 + item_ct1.get_local_id(0)) };	 //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")
		voxel = voxel + tetMinVoxel;

		if (voxel.x <= tetMaxVoxel.x && voxel.y <= tetMaxVoxel.y && voxel.z <= tetMaxVoxel.z)
		{
			vec3T<Tf> voxelPos = static_cast<vec3T<Tf>>(voxel) * resolution + bbMin;
			ScanConverterInternals::computeParametersVoxel3D(
				sampleDistance,
				scanlineLayout,
				scanlineIdxX, scanlineIdxY,
				s1, e1, s2, e2, s3, e3, s4, e4,
				scanline1Pos,
				scanline1Dir,
				scanline2Pos,
				scanline2Dir,
				scanline3Pos,
				scanline3Dir,
				scanline4Pos,
				scanline4Dir,
				startDepth, endDepth,
				imageSize,
				voxel,
				voxelPos,
				maskBuf,
				sampleIdxBuf,
				weightXBuf,
				weightYBuf,
				weightZBuf);
		}
	}

	template <typename InputType, typename OutputType, typename WeightType, typename IndexType>
	void scanConvert2D(
		uint32_t numScanlines,
		uint32_t numSamples,
		uint32_t width,
		uint32_t height,
		const uint8_t* __restrict__ mask,
		const IndexType* __restrict__ sampleIdx,
		const WeightType* __restrict__ weightX,
		const WeightType* __restrict__ weightY,
		const InputType* __restrict__ scanlines,
		OutputType* __restrict__ image,
		sycl::nd_item<3> item_ct1)
	{
		vec2T<uint32_t> pixelPos{ item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2),
								  item_ct1.get_local_range().get(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1) };  //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")

		if (pixelPos.x < width && pixelPos.y < height)
		{
			IndexType pixelIdx = pixelPos.x + pixelPos.y*width;
			float val = 0;
			if (mask[pixelIdx])
			{
				IndexType sIdx = sampleIdx[pixelIdx];
				WeightType wX = weightX[pixelIdx];
				WeightType wY = weightY[pixelIdx];

				val = (1 - wY)*((1 - wX)*scanlines[sIdx] +
					wX *scanlines[sIdx + 1]) +
					wY *((1 - wX)*scanlines[sIdx + numScanlines] +
						wX *scanlines[sIdx + 1 + numScanlines]);
			}
			image[pixelIdx] = clampCast<OutputType>(val);
		}
	}

	template <typename InputType, typename OutputType, typename WeightType, typename IndexType>
	void scanConvert3D(
		uint32_t numScanlinesX,
		uint32_t numScanlinesY,
		uint32_t numSamples,
		uint32_t width,
		uint32_t height,
		uint32_t depth,
		const uint8_t* __restrict__ mask,
		const IndexType* __restrict__ sampleIdx,
		const WeightType* __restrict__ weightX,
		const WeightType* __restrict__ weightY,
		const WeightType* __restrict__ weightZ,
		const InputType* __restrict__ scanlines,
		OutputType* __restrict__ image,
		sycl::nd_item<3> item_ct1)
	{
		vec3T<uint32_t> pixelPos{ item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2),
								  item_ct1.get_local_range().get(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1),
								  item_ct1.get_local_range().get(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0) };  //@suppress("Symbol is not resolved") @suppress("Field cannot be resolved")

		if (pixelPos.x < width && pixelPos.y < height)
		{
			IndexType pixelIdx = pixelPos.x + pixelPos.y*width + pixelPos.z*width*height;
			float val = 0;
			if (mask[pixelIdx])
			{
				uint32_t numScanlines = numScanlinesX*numScanlinesY;
				IndexType sIdx = sampleIdx[pixelIdx];
				WeightType wX = weightX[pixelIdx];
				WeightType wY = weightY[pixelIdx];
				WeightType wZ = weightZ[pixelIdx];


				val =
					(1 - wY)*((1 - wZ)*((1 - wX)*scanlines[sIdx] +
						wX *scanlines[sIdx + 1]) +
						wZ *((1 - wX)*scanlines[sIdx + numScanlinesX] +
							wX *scanlines[sIdx + 1 + numScanlinesX])) +
					wY* ((1 - wZ)*((1 - wX)*scanlines[sIdx + numScanlines] +
						wX *scanlines[sIdx + 1 + numScanlines]) +
						wZ *((1 - wX)*scanlines[sIdx + numScanlinesX + numScanlines] +
							wX *scanlines[sIdx + 1 + numScanlinesX + numScanlines]));
			}
			image[pixelIdx] = clampCast<OutputType>(val);
		}
	}

	shared_ptr<Container<uint8_t> > ScanConverter::getMask()
	{
		return m_mask;
	}

	template<typename InputType, typename OutputType>
	shared_ptr<Container<OutputType> > ScanConverter::convert(const shared_ptr<USImage>& inImage)
	{
		uint32_t numScanlines = (uint32_t)inImage->getImageProperties()->getNumScanlines();
		vec2s scanlineLayout = inImage->getImageProperties()->getScanlineLayout();
		uint32_t numSamples = (uint32_t)inImage->getImageProperties()->getNumSamples();

		shared_ptr<const Container<InputType> > pScanlineData = inImage->getData<InputType>();
		if (pScanlineData->isHost())
		{
			pScanlineData = make_shared<Container<InputType> >(LocationGpu, *pScanlineData);
		}
		auto pConv = make_shared<Container<OutputType> >(LocationGpu, pScanlineData->getStream(), m_imageSize.x*m_imageSize.y*m_imageSize.z);

		if (m_is2D)
		{
			sycl::range<3> blockSize(1, 256, 1);
			sycl::range<3> gridSize(1, static_cast<unsigned int>((m_imageSize.y + blockSize[ 1 ] - 1) / blockSize[ 1 ]),
									static_cast<unsigned int>((m_imageSize.x + blockSize[ 2 ] - 1) / blockSize[ 2 ]));
			/*
			DPCT1049:22: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
			*/
				  pScanlineData->getStream()->submit([ & ](sycl::handler& cgh) {
						auto m_imageSize_x_ct2 = ( uint32_t )m_imageSize.x;
						auto m_imageSize_y_ct3 = ( uint32_t )m_imageSize.y;
						auto m_mask_get_ct4 = m_mask->get();
						auto m_sampleIdx_get_ct5 = m_sampleIdx->get();
						auto m_weightX_get_ct6 = m_weightX->get();
						auto m_weightY_get_ct7 = m_weightY->get();
						auto pScanlineData_get_ct8 = pScanlineData->get();
						auto pConv_get_ct9 = pConv->get();

						cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), [ = ](sycl::nd_item<3> item_ct1) {
							  scanConvert2D(numScanlines, numSamples, m_imageSize_x_ct2, m_imageSize_y_ct3, m_mask_get_ct4, m_sampleIdx_get_ct5, m_weightX_get_ct6, m_weightY_get_ct7,
											pScanlineData_get_ct8, pConv_get_ct9, item_ct1);
						});
				  });
			/*
			DPCT1010:20: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
			*/
			cudaSafeCall(0);
		}
		else
		{
			sycl::range<3> blockSize(1, 256, 1);
			sycl::range<3> gridSize(static_cast<unsigned int>((m_imageSize.z + blockSize[ 0 ] - 1) / blockSize[ 0 ]), static_cast<unsigned int>((m_imageSize.y + blockSize[ 1 ] - 1) / blockSize[ 1 ]),
									static_cast<unsigned int>((m_imageSize.x + blockSize[ 2 ] - 1) / blockSize[ 2 ]));
			/*
			DPCT1049:23: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
			*/
				  pScanlineData->getStream()->submit([ & ](sycl::handler& cgh) {
						auto m_imageSize_x_ct3 = ( uint32_t )m_imageSize.x;
						auto m_imageSize_y_ct4 = ( uint32_t )m_imageSize.y;
						auto m_imageSize_z_ct5 = ( uint32_t )m_imageSize.z;
						auto m_mask_get_ct6 = m_mask->get();
						auto m_sampleIdx_get_ct7 = m_sampleIdx->get();
						auto m_weightX_get_ct8 = m_weightX->get();
						auto m_weightY_get_ct9 = m_weightY->get();
						auto m_weightZ_get_ct10 = m_weightZ->get();
						auto pScanlineData_get_ct11 = pScanlineData->get();
						auto pConv_get_ct12 = pConv->get();

						cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), [ = ](sycl::nd_item<3> item_ct1) {
							  scanConvert3D(( uint32_t )scanlineLayout.x, ( uint32_t )scanlineLayout.y, numSamples, m_imageSize_x_ct3, m_imageSize_y_ct4, m_imageSize_z_ct5, m_mask_get_ct6,
											m_sampleIdx_get_ct7, m_weightX_get_ct8, m_weightY_get_ct9, m_weightZ_get_ct10, pScanlineData_get_ct11, pConv_get_ct12, item_ct1);
						});
				  });
			/*
			DPCT1010:21: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
			*/
			cudaSafeCall(0);
		}
		return pConv;
	}

	template
		std::shared_ptr<Container<uint8_t> > ScanConverter::convert<uint8_t, uint8_t>(const std::shared_ptr<USImage>& inImage);
	template
		std::shared_ptr<Container<int16_t> > ScanConverter::convert<uint8_t, int16_t>(const std::shared_ptr<USImage>& inImage);
	template
		std::shared_ptr<Container<float> > ScanConverter::convert<uint8_t, float>(const std::shared_ptr<USImage>& inImage);
	template
		std::shared_ptr<Container<uint8_t> > ScanConverter::convert<int16_t, uint8_t>(const std::shared_ptr<USImage>& inImage);
	template
		std::shared_ptr<Container<int16_t> > ScanConverter::convert<int16_t, int16_t>(const std::shared_ptr<USImage>& inImage);
	template
		std::shared_ptr<Container<float> > ScanConverter::convert<int16_t, float>(const std::shared_ptr<USImage>& inImage);
	template
		std::shared_ptr<Container<uint8_t> > ScanConverter::convert<float, uint8_t>(const std::shared_ptr<USImage>& inImage);
	template
		std::shared_ptr<Container<int16_t> > ScanConverter::convert<float, int16_t>(const std::shared_ptr<USImage>& inImage);
	template
		std::shared_ptr<Container<float> > ScanConverter::convert<float, float>(const std::shared_ptr<USImage>& inImage);

	void ScanConverter::updateInternals(const std::shared_ptr<const USImageProperties>& inImageProps)
	{
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue&	  q_ct1 = dev_ct1.default_queue();
		logging::log_log("Scanconverter: Updating scanconversion internals");

		//Check the scanline configuration for validity
		m_is2D = inImageProps->is2D();

		vec2s layout = inImageProps->getScanlineLayout();
		double startDepth = 0;
		double endDepth = inImageProps->getDepth();
		double resolution = inImageProps->getImageResolution();
		auto scanlines = inImageProps->getScanlineInfo();
		logging::log_error_if(!scanlines, "ScanConverter: No scanlines have been attached to the USImageProperties!");

		bool scanlinesGood = scanlines.operator bool();

		if (scanlinesGood)
		{
			for (size_t scanlineIdxY = 0; scanlineIdxY < layout.y; scanlineIdxY++)
			{
				for (size_t scanlineIdxX = 0; scanlineIdxX < layout.x; scanlineIdxX++)
				{
					if (scanlineIdxX > 0)
					{
						vec start = (*scanlines)[scanlineIdxX][scanlineIdxY].getPoint(startDepth);
						vec startbefore = (*scanlines)[scanlineIdxX - 1][scanlineIdxY].getPoint(startDepth);
						vec end = (*scanlines)[scanlineIdxX][scanlineIdxY].getPoint(endDepth);
						vec endbefore = (*scanlines)[scanlineIdxX - 1][scanlineIdxY].getPoint(endDepth);

						//scanline start points are increasing in x
						scanlinesGood = scanlinesGood &&
							start.x >= startbefore.x;
						if (!scanlinesGood)
						{
							scanlinesGood = true;
							logging::log_error(":(  1 ", scanlineIdxX, ", ", scanlineIdxY);
						}
						//scanline end points are increasing in x, that means scanlines do not intersect
						scanlinesGood = scanlinesGood &&
							end.x >= endbefore.x;
						if (!scanlinesGood)
						{
							scanlinesGood = true;
							logging::log_error(":(  2 ", scanlineIdxX, ", ", scanlineIdxY);
						}
						//scanlines can not be identical
						scanlinesGood = scanlinesGood &&
							(start.x > startbefore.x || end.x > endbefore.x);
						if (!scanlinesGood)
						{
							scanlinesGood = true;
							logging::log_error(":(  3 ", scanlineIdxX, ", ", scanlineIdxY);
						}
						//scanlines are not skew
						scanlinesGood = scanlinesGood &&
							abs(det(start - endbefore, startbefore - endbefore, end - endbefore)) < m_skewnessTestThreshold;
						if (!scanlinesGood)
						{
							scanlinesGood = true;
							logging::log_error(":(  4 ", scanlineIdxX, ", ", scanlineIdxY);
						}
					}

					if (scanlineIdxY > 0)
					{
						vec start = (*scanlines)[scanlineIdxX][scanlineIdxY].getPoint(startDepth);
						vec startbefore = (*scanlines)[scanlineIdxX][scanlineIdxY - 1].getPoint(startDepth);
						vec end = (*scanlines)[scanlineIdxX][scanlineIdxY].getPoint(endDepth);
						vec endbefore = (*scanlines)[scanlineIdxX][scanlineIdxY - 1].getPoint(endDepth);

						//scanline start points are increasing in z
						scanlinesGood = scanlinesGood &&
							start.z >= startbefore.z;
						if (!scanlinesGood)
						{
							scanlinesGood = true;
							logging::log_error(":(  5 ", scanlineIdxX, ", ", scanlineIdxY);
						}
						//scanline end points are increasing in z, that means scanlines do not intersect
						scanlinesGood = scanlinesGood &&
							end.z >= endbefore.z;
						if (!scanlinesGood)
						{
							scanlinesGood = true;
							logging::log_error(":(  6 ", scanlineIdxX, ", ", scanlineIdxY);
						}
						//scanlines can not be identical
						scanlinesGood = scanlinesGood &&
							(start.z > startbefore.z || end.z > endbefore.z);
						if (!scanlinesGood)
						{
							scanlinesGood = true;
							logging::log_error(":(  7 ", scanlineIdxX, ", ", scanlineIdxY);
						}
						//scanlines are not skew
						scanlinesGood = scanlinesGood &&
							abs(det(start - endbefore, startbefore - endbefore, end - endbefore)) < m_skewnessTestThreshold;
						if (!scanlinesGood)
						{
							scanlinesGood = true;
							logging::log_error(":(  8 ", scanlineIdxX, ", ", scanlineIdxY, "   det = ", det(start - endbefore, startbefore - endbefore, end - endbefore));
						}
					}
				}
			}
		}

		if (scanlinesGood)
		{
			//find scan bounding box
			vec bbMin{ numeric_limits<double>::max(),  numeric_limits<double>::max(),  numeric_limits<double>::max() };
			vec bbMax{ -numeric_limits<double>::max(), -numeric_limits<double>::max(), -numeric_limits<double>::max() };
			for (size_t scanlineIdxY = 0; scanlineIdxY < layout.y; scanlineIdxY++)
			{
				for (size_t scanlineIdxX = 0; scanlineIdxX < layout.x; scanlineIdxX++)
				{
					vec p1 = (*scanlines)[scanlineIdxX][scanlineIdxY].getPoint(startDepth);
					vec p2 = (*scanlines)[scanlineIdxX][scanlineIdxY].getPoint(endDepth);
					bbMin = { min(bbMin.x, p1.x), min(bbMin.y, p1.y), min(bbMin.z, p1.z) };
					bbMax = { max(bbMax.x, p1.x), max(bbMax.y, p1.y), max(bbMax.z, p1.z) };
					bbMin = { min(bbMin.x, p2.x), min(bbMin.y, p2.y), min(bbMin.z, p2.z) };
					bbMax = { max(bbMax.x, p2.x), max(bbMax.y, p2.y), max(bbMax.z, p2.z) };
				}
			}
			m_bbMin = bbMin;
			m_bbMax = bbMax;

			//compute image size
			m_imageSize = static_cast<vec3s>(ceil((bbMax - bbMin) / resolution)) + 1;
			m_imageSize.x = max(m_imageSize.x, (size_t)1);
			m_imageSize.y = max(m_imageSize.y, (size_t)1);
			m_imageSize.z = max(m_imageSize.z, (size_t)1);

			// create buffers
			size_t numelBuffers = m_imageSize.x*m_imageSize.y*m_imageSize.z;
			/*m_mask = make_shared<Container<uint8_t> >(ContainerLocation::LocationHost, numelBuffers);
			m_sampleIdx = make_shared<Container<IndexType> >(ContainerLocation::LocationHost, numelBuffers);
			m_weightX = make_shared<Container<WeightType> >(ContainerLocation::LocationHost, numelBuffers);
			m_weightY = make_shared<Container<WeightType> >(ContainerLocation::LocationHost, numelBuffers);
			m_weightZ = make_shared<Container<WeightType> >(ContainerLocation::LocationHost, numelBuffers);*/
			m_mask = make_shared<Container<uint8_t>>(ContainerLocation::LocationGpu, &q_ct1, numelBuffers);
			m_sampleIdx = make_shared<Container<IndexType>>(ContainerLocation::LocationGpu, &q_ct1, numelBuffers);
			m_weightX = make_shared<Container<WeightType>>(ContainerLocation::LocationGpu, &q_ct1, numelBuffers);
			m_weightY = make_shared<Container<WeightType>>(ContainerLocation::LocationGpu, &q_ct1, numelBuffers);
			m_weightZ = make_shared<Container<WeightType>>(ContainerLocation::LocationGpu, &q_ct1, numelBuffers);

			//create image mask
			/*
			DPCT1003:24: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
			*/
			cudaSafeCall((q_ct1.memset(m_mask->get(), 0, m_mask->size() * sizeof(uint8_t)), 0));

			if (m_is2D)
			{
				//2D is computed on the cpu at the moment -> copy
				m_mask = make_shared<Container<uint8_t> >(LocationHost, *m_mask);
				m_sampleIdx = make_shared<Container<IndexType> >(LocationHost, *m_sampleIdx);
				m_weightX = make_shared<Container<WeightType> >(LocationHost, *m_weightX);
				m_weightY = make_shared<Container<WeightType> >(LocationHost, *m_weightY);
				m_weightZ = make_shared<Container<WeightType> >(LocationHost, *m_weightZ);

				vec2 bb2DMin{ m_bbMin.x, m_bbMin.y };
				assert(layout.x > 1);
				// From now on, we assume everything is in the xy-plane
				// -----------------------------------------
				for (size_t scanlineIdxY = 0; scanlineIdxY < layout.y; scanlineIdxY++)
				{
#pragma omp parallel for schedule(dynamic, 8)
					for (int scanlineIdxX = 0; scanlineIdxX < layout.x - 1; scanlineIdxX++)
					{
						vec start3 = (*scanlines)[scanlineIdxX][scanlineIdxY].getPoint(startDepth);
						vec startN3 = (*scanlines)[scanlineIdxX + 1][scanlineIdxY].getPoint(startDepth);
						vec end3 = (*scanlines)[scanlineIdxX][scanlineIdxY].getPoint(endDepth);
						vec endN3 = (*scanlines)[scanlineIdxX + 1][scanlineIdxY].getPoint(endDepth);
						vec2 start = { start3.x, start3.y };
						vec2 startN = { startN3.x, startN3.y };
						vec2 end = { end3.x, end3.y };
						vec2 endN = { endN3.x, endN3.y };

						// find bounding box of the two scanlines
						vec2 quadMinWorld = { std::min(std::min(std::min(start.x, startN.x), end.x), endN.x), std::min(std::min(std::min(start.y, startN.y), end.y), endN.y) };
						vec2 quadMaxWorld = { std::max(std::max(std::max(start.x, startN.x), end.x), endN.x), std::max(std::max(std::max(start.y, startN.y), end.y), endN.y) };

						vec2s quadMinPixel = static_cast<vec2s>(floor((quadMinWorld - bb2DMin) / resolution));
						vec2s quadMaxPixel = static_cast<vec2s>(ceil((quadMaxWorld - bb2DMin) / resolution));

						// check the pixels in the quad bounding box and mark the inside ones
						vec2s pixel;
						for (pixel.x = quadMinPixel.x; pixel.x <= quadMaxPixel.x; pixel.x++)
						{
							for (pixel.y = quadMinPixel.y; pixel.y <= quadMaxPixel.y; pixel.y++)
							{
								vec2 pixelPos = static_cast<vec2>(pixel) * resolution + bb2DMin;
								if (pointInsideTriangle(endN, end, start, pixelPos) ||
									pointInsideTriangle(start, startN, endN, pixelPos))
								{
									m_mask->get()[pixel.x + pixel.y*m_imageSize.x] = 1;

									vec2 params = mapToParameters2D(
										(*scanlines)[scanlineIdxX][scanlineIdxY].position,
										(*scanlines)[scanlineIdxX + 1][scanlineIdxY].position,
										(*scanlines)[scanlineIdxX][scanlineIdxY].direction,
										(*scanlines)[scanlineIdxX + 1][scanlineIdxY].direction,
										startDepth, endDepth, { pixelPos.x, pixelPos.y, 0.0 });
									double t = params.x;
									double d = params.y;

									IndexType  sampleIdxScanline = static_cast<IndexType>(std::floor(d / inImageProps->getSampleDistance()));
									WeightType weightY = static_cast<WeightType>(d - (sampleIdxScanline*inImageProps->getSampleDistance()));
									WeightType weightX = static_cast<WeightType>(t);

									IndexType sampleIdx = static_cast<IndexType>(sampleIdxScanline*inImageProps->getNumScanlines() +
										scanlineIdxX + scanlineIdxY*layout.x);

									m_sampleIdx->get()[pixel.x + pixel.y*m_imageSize.x] = sampleIdx;
									m_weightX->get()[pixel.x + pixel.y*m_imageSize.x] = weightX;
									m_weightY->get()[pixel.x + pixel.y*m_imageSize.x] = weightY;
								}
							}
						}
					}
				}

				//2D is computed on the cpu at the moment -> copy
				m_mask = make_shared<Container<uint8_t> >(LocationGpu, *m_mask);
				m_sampleIdx = make_shared<Container<IndexType> >(LocationGpu, *m_sampleIdx);
				m_weightX = make_shared<Container<WeightType> >(LocationGpu, *m_weightX);
				m_weightY = make_shared<Container<WeightType> >(LocationGpu, *m_weightY);
				m_weightZ = make_shared<Container<WeightType> >(LocationGpu, *m_weightZ);
			}
			else {
				// 3D case
				for (int scanlineIdxY = 0; scanlineIdxY < layout.y - 1; scanlineIdxY++)
				{
					//#pragma omp parallel for schedule(dynamic, 1)
					for (int scanlineIdxX = 0; scanlineIdxX < layout.x - 1; scanlineIdxX++)
					{
						vec s1 = (*scanlines)[scanlineIdxX][scanlineIdxY].getPoint(startDepth);
						vec e1 = (*scanlines)[scanlineIdxX][scanlineIdxY].getPoint(endDepth);
						vec s2 = (*scanlines)[scanlineIdxX + 1][scanlineIdxY].getPoint(startDepth);
						vec e2 = (*scanlines)[scanlineIdxX + 1][scanlineIdxY].getPoint(endDepth);
						vec s3 = (*scanlines)[scanlineIdxX][scanlineIdxY + 1].getPoint(startDepth);
						vec e3 = (*scanlines)[scanlineIdxX][scanlineIdxY + 1].getPoint(endDepth);
						vec s4 = (*scanlines)[scanlineIdxX + 1][scanlineIdxY + 1].getPoint(startDepth);
						vec e4 = (*scanlines)[scanlineIdxX + 1][scanlineIdxY + 1].getPoint(endDepth);

						// find bounding box of the four scanlines
						vec tetMinWorld = min(min(min(s1, s2), min(s3, s4)),
							min(min(e1, e2), min(e3, e4)));
						vec tetMaxWorld = max(max(max(s1, s2), max(s3, s4)),
							max(max(e1, e2), max(e3, e4)));

						vec3s tetMinVoxel = static_cast<vec3s>(floor((tetMinWorld - bbMin) / resolution));
						vec3s tetMaxVoxel = static_cast<vec3s>(ceil((tetMaxWorld - bbMin) / resolution));

						vec3s numVoxels = tetMaxVoxel - tetMinVoxel + 1;
						sycl::range<3> blockSize(4, 4, 16);
						sycl::range<3> gridSize(static_cast<unsigned int>((numVoxels.z + blockSize[ 0 ] - 1) / blockSize[ 0 ]),
												static_cast<unsigned int>((numVoxels.y + blockSize[ 1 ] - 1) / blockSize[ 1 ]),
												static_cast<unsigned int>((numVoxels.x + blockSize[ 2 ] - 1) / blockSize[ 2 ]));

						typedef float Tf;
						typedef int Ti;

						/*
						DPCT1049:25: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if
						needed.
						*/
									q_ct1.submit([ & ](sycl::handler& cgh) {
										  auto inImageProps_getSampleDistance_ct0 = static_cast<Tf>(inImageProps->getSampleDistance());
										  auto static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_position_ct12 = static_cast<vec3T<Tf>>((*scanlines)[ scanlineIdxX ][ scanlineIdxY ].position);
										  auto static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_direction_ct13 = static_cast<vec3T<Tf>>((*scanlines)[ scanlineIdxX ][ scanlineIdxY ].direction);
										  auto static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_position_ct14 =
											  static_cast<vec3T<Tf>>((*scanlines)[ scanlineIdxX + 1 ][ scanlineIdxY ].position);
										  auto static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_direction_ct15 =
											  static_cast<vec3T<Tf>>((*scanlines)[ scanlineIdxX + 1 ][ scanlineIdxY ].direction);
										  auto static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_position_ct16 =
											  static_cast<vec3T<Tf>>((*scanlines)[ scanlineIdxX ][ scanlineIdxY + 1 ].position);
										  auto static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_direction_ct17 =
											  static_cast<vec3T<Tf>>((*scanlines)[ scanlineIdxX ][ scanlineIdxY + 1 ].direction);
										  auto static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_position_ct18 =
											  static_cast<vec3T<Tf>>((*scanlines)[ scanlineIdxX + 1 ][ scanlineIdxY + 1 ].position);
										  auto static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_direction_ct19 =
											  static_cast<vec3T<Tf>>((*scanlines)[ scanlineIdxX + 1 ][ scanlineIdxY + 1 ].direction);
										  auto static_cast_vec3T_Ti_m_imageSize_ct22 = static_cast<vec3T<Ti>>(m_imageSize);
										  auto m_mask_get_ct27 = m_mask->get();
										  auto m_sampleIdx_get_ct28 = m_sampleIdx->get();
										  auto m_weightX_get_ct29 = m_weightX->get();
										  auto m_weightY_get_ct30 = m_weightY->get();
										  auto m_weightZ_get_ct31 = m_weightZ->get();

										  cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize), [ = ](sycl::nd_item<3> item_ct1) {
												computeParameterBB3D<Tf, Ti>(
													inImageProps_getSampleDistance_ct0, static_cast<vec2T<Ti>>(layout), scanlineIdxX, scanlineIdxY, static_cast<vec3T<Tf>>(s1),
													static_cast<vec3T<Tf>>(e1), static_cast<vec3T<Tf>>(s2), static_cast<vec3T<Tf>>(e2), static_cast<vec3T<Tf>>(s3), static_cast<vec3T<Tf>>(e3),
													static_cast<vec3T<Tf>>(s4), static_cast<vec3T<Tf>>(e4), static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_position_ct12,
													static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_direction_ct13, static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_position_ct14,
													static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_direction_ct15, static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_position_ct16,
													static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_direction_ct17, static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_position_ct18,
													static_cast_vec3T_Tf_scanlines_scanlineIdxX_scanlineIdxY_direction_ct19, static_cast<Tf>(startDepth), static_cast<Tf>(endDepth),
													static_cast_vec3T_Ti_m_imageSize_ct22, static_cast<vec3T<Tf>>(bbMin), static_cast<vec3T<Ti>>(tetMinVoxel), static_cast<vec3T<Ti>>(tetMaxVoxel),
													static_cast<Tf>(resolution), m_mask_get_ct27, m_sampleIdx_get_ct28, m_weightX_get_ct29, m_weightY_get_ct30, m_weightZ_get_ct31, item_ct1);
										  });
									});
						/*
						DPCT1010:26: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
						*/
						cudaSafeCall(0);
					}
				}
			}
			/*
			DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
			*/
			cudaSafeCall((q_ct1.wait(), 0));
		}
		else
		{
			logging::log_error("ScanConverter: The scanlines are not in the required configuration.");
		}
	}

	double ScanConverter::barycentricCoordinate2D(const vec2 & a, const vec2 & b, const vec2 & c)
	{
		return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
	}

	bool ScanConverter::pointInsideTriangle(const vec2 & a, const vec2 & b, const vec2 & c, const vec2 & p)
	{
		double w0 = barycentricCoordinate2D(b, c, p);
		double w1 = barycentricCoordinate2D(c, a, p);
		double w2 = barycentricCoordinate2D(a, b, p);

		// Test if p is on or inside all edges
		return (w0 >= 0 && w1 >= 0 && w2 >= 0);
	}

	vec ScanConverter::pointLineConnection(const vec & a, const vec & da, const vec & x)
	{
		vec conn = x - a;
		vec r = conn - dot(da, conn) *da;
		return r;
	}

	vec2 ScanConverter::mapToParameters2D(const vec & a, const vec & b, const vec & da, const vec & db, double startDepth, double endDepth, const vec & x)
	{
		//find t via binary search
		double lowT = 0;
		double highT = 1;
		vec lowConn = pointLineConnection(a, da, x);
		vec highConn = pointLineConnection(b, db, x);
		double lowDist = norm(lowConn);
		double highDist = norm(highConn);

		if (highConn.x == 0 && highConn.y == 0 && highConn.z == 0)
		{
			double t = highT;
			double d = norm(x - b);
			return{ t, d };
		}
		else if (lowConn.x == 0 && lowConn.y == 0 && lowConn.z == 0)
		{
			double t = lowT;
			double d = norm(x - a);
			return{ t, d };
		}

		assert(dot(lowConn, highConn) < 0);

		double dist = 1e10;
		double t = (highT - lowT) / 2 + lowT;
		vec lineBase;
		for (size_t numIter = 0; numIter < m_mappingMaxIterations && dist > m_mappingDistanceThreshold; numIter++)
		{
			t = (1 - highDist / (highDist + lowDist))*highT + (1 - lowDist / (highDist + lowDist))*lowT;

			lineBase = (1 - t)*a + t*b;
			vec lineDir = slerp3(da, db, t);

			vec conn = pointLineConnection(lineBase, lineDir, x);
			dist = norm(conn);

			if (dot(lowConn, conn) < 0)
			{
				highT = t;
				highConn = conn;
				highDist = dist;
			}
			else
			{
				lowT = t;
				lowConn = conn;
				lowDist = dist;
			}
		}
		double d = norm(x - lineBase);

		return{ t, d };
	}
}
