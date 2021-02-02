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

#ifndef __WINDOWFUNCTION_H__
#define __WINDOWFUNCTION_H__

#ifndef SYCL_LANGUAGE_VERSION
#include <algorithm>
#endif

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <memory>
#include <Container.h>
#include <utilities/utility.h>
#include <utilities/cudaUtility.h>
#include <cmath>

namespace supra
{
#ifndef SYCL_LANGUAGE_VERSION
	using std::max;
	using std::min;
#else
	using sycl::max;
	using sycl::min;
#endif

	//forward declaration
	class WindowFunction;

	enum WindowType : uint32_t
	{
		WindowRectangular = 0,
		WindowHann = 1,
		WindowHamming = 2,
		WindowGauss = 3,
		WindowINVALID = 4
	};

	class WindowFunctionGpu
	{
	public:
		typedef float ElementType;

		WindowFunctionGpu(const WindowFunctionGpu& a)
			: m_numEntriesPerFunction(a.m_numEntriesPerFunction)
			, m_data(a.m_data)
			, m_scale(a.m_scale) {};

		//Returns the weight of chosen window a the relative index 
		// relativeIndex has to be normalized to [-1, 1] (inclusive)
		inline ElementType get(float relativeIndex) const
		{
			float	 relativeIndexClamped = sycl::min(sycl::max(relativeIndex, -1.0f), 1.0f);
			uint32_t absoluteIndex = static_cast<uint32_t>(sycl::round(m_scale * (relativeIndexClamped + 1.0f)));
			return m_data[absoluteIndex];
		}

		//Returns the weight of chosen window a the relative index
		// relativeIndex has to be normalized to [-1, 1] (inclusive)
		inline ElementType getShared(const ElementType * __restrict__ sharedData, float relativeIndex) const
		{
			float	 relativeIndexClamped = sycl::min(sycl::max(relativeIndex, -1.0f), 1.0f);
			uint32_t absoluteIndex = static_cast<uint32_t>(sycl::round(m_scale * (relativeIndexClamped + 1.0f)));
			return sharedData[absoluteIndex];
		}

		inline ElementType getDirect(uint32_t idx) const
		{
			ElementType ret = 0;
			if (idx < m_numEntriesPerFunction)
			{
				ret = m_data[idx];
			}
			return ret;
		}

		inline uint32_t numElements() const
		{
			return m_numEntriesPerFunction;
		}

	private:
		friend WindowFunction;
		WindowFunctionGpu(size_t numEntriesPerFunction, const ElementType* data)
			: m_numEntriesPerFunction(static_cast<uint32_t>(numEntriesPerFunction))
			, m_data(data)
			, m_scale(static_cast<float>(numEntriesPerFunction - 1)*0.5f) {};

		float m_scale;
		uint32_t m_numEntriesPerFunction;
		const ElementType* m_data;
	};

	class WindowFunction
	{
	public:
		typedef WindowFunctionGpu::ElementType ElementType;

		WindowFunction(WindowType type, ElementType windowParameter = 0.0, size_t numEntriesPerFunction = 128);

		const WindowFunctionGpu* getGpu() const;

		WindowType getType() const { return m_type; };
		ElementType getParameter() const { return m_windowParameter; };

		//Returns the weight of chosen window a the relative index
		// relativeIndex has to be normalized to [-1, 1] (inclusive)
		ElementType get(float relativeIndex) const;
		ElementType getDirect(uint32_t idx) const;

		// relativeIndex has to be normalized to [-1, 1] (inclusive)
		template <typename T>
		static inline T windowFunction(const WindowType& type, const T& relativeIndex, const T& windowParameter)
		{
			switch (type)
			{
			case WindowRectangular:
				return 1.0;
			case WindowHann:
				return (1 - windowParameter)*(0.5f - 0.5f*std::cos(2*static_cast<T>(M_PI)*((relativeIndex + 1) *0.5f))) + windowParameter;
			case WindowHamming:
				return (1 - windowParameter)*(0.54f - 0.46f*std::cos(2*static_cast<T>(M_PI)*((relativeIndex + 1) *0.5f))) + windowParameter;
			case WindowGauss:
				return static_cast<T>(1.0 / (windowParameter * sycl::sqrt(2.0 * M_PI)) * std::exp((-1.0 / 2.0) * (relativeIndex / windowParameter) * (relativeIndex / windowParameter)));
			default:
				return 0;
			}
		}
	private:
		size_t m_numEntriesPerFunction;
		std::vector<ElementType> m_data;
		std::unique_ptr<Container<ElementType> > m_dataGpu;
		ElementType m_scale;
		WindowType m_type;
		ElementType m_windowParameter;
		WindowFunctionGpu m_gpuFunction;
	};
}

#endif //!__WINDOWFUNCTION_H__
