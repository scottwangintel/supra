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

#ifndef __FIRFILTERFACTORY_H__
#define __FIRFILTERFACTORY_H__

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <memory>
#include <functional>
#include "Container.h"
#include <cmath>

namespace supra
{
	/// A factory for FIR filters
	class FirFilterFactory {
	public:
		/// Enum for the different filter types
		enum FilterType {
			FilterTypeLowPass,
			FilterTypeHighPass,
			FilterTypeBandPass,
			FilterTypeHilbertTransformer
		};

		/// Enum for the different window types used in creating filters
		enum FilterWindow {
			FilterWindowRectangular,
			FilterWindowHann,
			FilterWindowHamming,
			FilterWindowKaiser
		};

		/// Returns a FIR filter constructed with the window-method
		template <typename ElementType>
		static std::shared_ptr<Container<ElementType> >
			createFilter(const size_t &length, const FilterType &type, const FilterWindow &window, const double &samplingFrequency = 2.0, const double &frequency = 0.0, const double &bandwidth = 0.0)
		{
			std::shared_ptr<Container<ElementType> > filter = createFilterNoWindow<ElementType>(length, type, samplingFrequency, frequency, bandwidth);
			applyWindowToFilter<ElementType>(filter, window);
			if (type == FilterTypeBandPass)
			{
				normalizeGain<ElementType>(filter, samplingFrequency, frequency);
			}

			return filter;
		}

	private:
		template <typename ElementType>
		static std::shared_ptr<Container<ElementType> >
			createFilterNoWindow(const size_t &length, const FilterType &type, const double &samplingFrequency, const double &frequency, const double &bandwidth)
		{
			if (type == FilterTypeHighPass || type == FilterTypeBandPass || type == FilterTypeLowPass)
			{
				assert(samplingFrequency != 0.0);
				assert(frequency != 0.0);
			}
			if (type == FilterTypeBandPass)
			{
				assert(bandwidth != 0.0);
			}

			ElementType omega = static_cast<ElementType>(2 * M_PI* frequency / samplingFrequency);
			ElementType omegaBandwidth = static_cast<ElementType>(2 * M_PI* bandwidth / samplingFrequency);
			int halfWidth = ((int)length - 1) / 2;

			sycl::queue &default_queue=dpct::get_default_queue();
			auto filter = std::make_shared<Container<ElementType>>(LocationHost, &default_queue, length);

			//determine the filter function
			std::function<ElementType(int)> filterFunction = [&halfWidth](int n) -> ElementType {
				if (n == halfWidth)
				{
					return static_cast<ElementType>(1);
				}
				else {
					return static_cast<ElementType>(0);
				}
			};
			switch (type)
			{
			case FilterTypeHilbertTransformer:
				// Following formula 2 in
				// "Carrick, Matt, and Doug Jaeger. "Design and Application of a Hilbert Transformer in a Digital Receiver." (2011)."
				filterFunction = [halfWidth](int n) -> ElementType {
					auto k = (n - halfWidth);
					if (k % 2 == 0)
					{
						return static_cast<ElementType>(0);
					}
					else
					{
						return static_cast<ElementType>(2.0 / (M_PI * k));
					}
				};
				break;
			case FilterTypeHighPass:
				filterFunction = [omega, halfWidth](int n) -> ElementType {
					if (n == halfWidth)
					{
						return static_cast<ElementType>(1 - omega / M_PI);
					}
					else {
						return static_cast<ElementType>(-omega / M_PI * sin(omega * (n - halfWidth)) / (omega * (n - halfWidth)));
					}
				};
				break;
			case FilterTypeBandPass:
				filterFunction = [omega, omegaBandwidth, halfWidth](int n) -> ElementType {
					if (n == halfWidth)
					{
						return static_cast<ElementType>(2.0 * omegaBandwidth / M_PI);
					}
					else {
						return static_cast<ElementType>(
							2.0 * cos(omega * n - halfWidth) *
							omegaBandwidth / M_PI * sin(omegaBandwidth * (n - halfWidth)) / (omegaBandwidth * (n - halfWidth)));
					}
				};
				break;
			case FilterTypeLowPass:
			default:
				filterFunction = [omega, halfWidth](int n) -> ElementType {
					if (n == halfWidth)
					{
						return static_cast<ElementType>(omega / M_PI);
					}
					else {
						return static_cast<ElementType>(omega / M_PI * sin(omega * (n - halfWidth)) / (omega * (n - halfWidth)));
					}
				};
				break;
			}

			//create the filter
			for (size_t k = 0; k < length; k++)
			{
				filter->get()[k] = filterFunction((int)k);
			}

			return filter;
		}

		template <typename ElementType>
		static void applyWindowToFilter(std::shared_ptr<Container<ElementType> > filter, FilterWindow window)
		{
			size_t filterLength = filter->size();
			size_t maxN = filterLength - 1;
			ElementType beta = (ElementType)4.0;
			std::function<ElementType(int)> windowFunction = [filterLength](int n) -> ElementType { return static_cast<ElementType>(1); };
			switch (window)
			{
			case FilterWindowHann:
				windowFunction = [maxN](int n) -> ElementType { return static_cast<ElementType>(
					0.50 - 0.50*cos(2 * M_PI * n / maxN)); };
				break;
			case FilterWindowHamming:
				windowFunction = [maxN](int n) -> ElementType { return static_cast<ElementType>(
					0.54 - 0.46*cos(2 * M_PI * n / maxN)); };
				break;
			case FilterWindowKaiser:
				windowFunction = [maxN, beta](int n) -> ElementType {
					double argument = beta * sycl::sqrt(1.0 - (2 * (( ElementType )n - maxN / 2) / maxN) * (2 * (( ElementType )n - maxN / 2) / maxN));
					return static_cast<ElementType>(bessel0_1stKind(argument) / bessel0_1stKind(beta)); };
				break;
			case FilterWindowRectangular:
			default:
				windowFunction = [](int n) -> ElementType { return static_cast<ElementType>(1); };
				break;
			}

			for (size_t k = 0; k < filterLength; k++)
			{
				filter->get()[k] *= windowFunction((int)k);
			}
		}

		template <typename ElementType>
		static void normalizeGain(std::shared_ptr<Container<ElementType> > filter, double samplingFrequency, double frequency)
		{
			ElementType omega = static_cast<ElementType>(2 * M_PI* frequency / samplingFrequency);
			ElementType gainR = 0;
			ElementType gainI = 0;

			for (int k = 0; k < filter->size(); k++)
			{
				gainR += filter->get()[k] * cos(omega * (ElementType)k);
				gainI += filter->get()[k] * sin(omega * (ElementType)k);
			}
			ElementType gain = sycl::sqrt(gainR*gainR + gainI*gainI);
			for (int k = 0; k < filter->size(); k++)
			{
				filter->get()[k] /= gain;
			}
		}

		template <typename T>
		static T bessel0_1stKind(const T &x)
		{
			T sum = 0.0;
			//implemented look up factorial. 
			static const int factorial[9] = { 1, 2, 6, 24, 120, 720, 5040, 40320, 362880 };
			for (int k = 1; k < 10; k++)
			{
				T xPower = pow(x / ( T )2.0, ( T )k);
				// 1, 2, 6, 24, 120, 720, 5040, 40320, 362880
				sum += pow(xPower / ( T )factorial[ k - 1 ], ( T )2.0);
			}
			return (T)1.0 + sum;
		}
	};
}

#endif // !__FIRFILTERFACTORY_H__
