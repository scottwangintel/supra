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

#ifndef __CONTAINER_H__
#define __CONTAINER_H__

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ContainerFactory.h"
#ifdef HAVE_CUDA
#include "utilities/syclUtility.h"
#endif
#include "utilities/DataType.h"

#include <exception>
#include <memory>
#include <vector>
#include <cassert>
#include <chrono>

#include <future>

namespace supra
{
	class ContainerBase
	{
	public:
		virtual ~ContainerBase() {};
		virtual DataType getType() const { return TypeUnknown; };
	};

	template<typename T>
	class Container : public ContainerBase
	{
	public:
		typedef ContainerFactory::ContainerStreamType ContainerStreamType;

		Container(ContainerLocation location, ContainerStreamType associatedStream, size_t numel)
		{
#ifndef HAVE_CUDA
			location = LocationHost;
#endif

			m_numel = numel;
			m_location = location;
			m_associatedStream = associatedStream;

			m_buffer = reinterpret_cast<T*>(ContainerFactoryContainerInterface::acquireMemory(
				m_numel * sizeof(T), m_location));
		};
		Container(ContainerLocation location, ContainerStreamType associatedStream, const std::vector<T> & data, bool waitFinished = true)
			:Container(location, associatedStream, data.size())
		{
#ifdef HAVE_CUDA
			if(location == LocationGpu)
			{
				
				associatedStream->memcpy(this->get(), data.data(), this->size() * sizeof(T));
				associatedStream->wait();
				
			}
			else if(location == LocationBoth)
			{
				
				associatedStream->memcpy(this->get(), data.data(), this->size() * sizeof(T));
				associatedStream->wait();
			}
			else
			{
				std::copy(data.begin(), data.end(), this->get());
			}
			if (waitFinished)
			{
				waitCreationFinished();
			}
#else
			std::copy(data.begin(), data.end(), this->get());
#endif
		};
		Container(ContainerLocation location, ContainerStreamType associatedStream, const T* dataBegin, const T* dataEnd, bool waitFinished = true)
			:Container(location, associatedStream, dataEnd - dataBegin)
		{
#ifdef HAVE_CUDA
			
			associatedStream->memcpy(this->get(), dataBegin, this->size() * sizeof(T));
			createAndRecordEvent();
			if (waitFinished)
			{
				waitCreationFinished();
			}
#else
			std::copy(dataBegin, dataEnd, this->get());
#endif
		};
		Container(ContainerLocation location, const Container<T>& source, bool waitFinished = true)
			: Container(location, source.getStream(), source.size())
		{
			if (source.m_location == LocationHost && location == LocationHost)
			{
				std::copy(source.get(), source.get() + source.size(), this->get());
			}
#ifdef HAVE_CUDA
			else if (source.m_location == LocationHost && location == LocationGpu)
			{
				
				source.getStream()->memcpy(this->get(), source.get(), source.size() * sizeof(T));
				source.getStream()->wait();
			}
			else if (source.m_location == LocationGpu && location == LocationHost)
			{
				
				source.getStream()->memcpy(this->get(), source.get(), source.size() * sizeof(T));
				source.getStream()->wait();
			}
			else if (source.m_location == LocationGpu && location == LocationGpu)
			{
				
				source.getStream()->memcpy(this->get(), source.get(), source.size() * sizeof(T));
				source.getStream()->wait();
			}
			else
			{
				
				source.getStream()->memcpy(this->get(), source.get(), source.size() * sizeof(T));
				source.getStream()->wait();
			}
			if (waitFinished)
			{
				waitCreationFinished();
			}
#else
			std::copy(source.get(), source.get() + source.size(), this->get());
#endif
		};
		~Container()
		 try {
#ifdef HAVE_CUDA
			
			auto ret = 0;
			if (ret != 0 && ret != 600 && ret != 4)
			{
				syclSafeCall(ret);
			}
			// If the driver is currently unloading, we cannot free the memory in any way. Exit will clean up.
			else if (ret != 4)
			{
				if (ret == 0)
				{
					ContainerFactoryContainerInterface::returnMemory(reinterpret_cast<uint8_t*>(m_buffer), m_numel * sizeof(T), m_location);
				}
				else
				{
					auto buffer = m_buffer;
					auto numel = m_numel;
					auto location = m_location;
					addCallbackStream([ buffer, numel, location ](sycl::queue* s, int e) -> void {
						ContainerFactoryContainerInterface::returnMemory(reinterpret_cast<uint8_t*>(buffer), numel * sizeof(T), location);
					});
				}
			}
#else
			ContainerFactoryContainerInterface::returnMemory(reinterpret_cast<uint8_t*>(m_buffer), m_numel * sizeof(T), m_location);
#endif
		}
		catch (sycl::exception const& exc) {
		  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
		  std::exit(1);
		};

		const T* get() const { return m_buffer; };
		T* get() { return m_buffer; };

		T* getCopyHostRaw() const
		{
#ifdef HAVE_CUDA
			auto ret = new T[this->size()];
			
			if(m_location == LocationHost)
			{
				std::copy(this->get(), this->get() + this->size(), ret);
			}
			else if(m_location == LocationGpu)
			{
				
				getStream()->memcpy(ret, this->get(), this->size() * sizeof(T));
				getStream()->wait();				
			}
			else 
			{
				
				dpct::get_default_queue().memcpy(ret, this->get(), this->size() * sizeof(T)).wait();
			}
			return ret;
#else
			return nullptr;
#endif
		}

		void copyTo(T* dst, size_t maxSize) const
		{
#ifdef HAVE_CUDA
			assert(maxSize >= this->size());
			
			dpct::get_default_queue().memcpy(dst, this->get(), this->size() * sizeof(T)).wait();
#endif
		}

		void waitCreationFinished()
		{
			m_associatedStream->wait();
		}

		// returns the number of elements that can be stored in this container
		size_t size() const { return m_numel; };

		bool isHost() const { return m_location == ContainerLocation::LocationHost; };
		bool isGPU() const { return m_location == ContainerLocation::LocationGpu; };
		bool isBoth() const { return m_location == ContainerLocation::LocationBoth; };
		ContainerLocation getLocation() const { return m_location; };
		ContainerStreamType getStream() const
		{
			return m_associatedStream;
		}
		DataType getType() const { return DataTypeGet<T>(); }

	private:
		void createAndRecordEvent()
		{

		}

#ifdef HAVE_CUDA
		void addCallbackStream(std::function<void(sycl::queue*, int)> func)
		{
			auto funcPointer = new std::function<void(sycl::queue*, int)>(func);
			
			std::async([ & ]() {
				m_associatedStream->wait(); 
				(Container<T>::cudaDeleteCallback)(m_associatedStream, 0, funcPointer);
						  });
		}
#endif

#ifdef HAVE_CUDA
		static void cudaDeleteCallback(sycl::queue* stream, int status, void* userData)
		{
			std::unique_ptr<std::function<void(sycl::queue*, int)>> func = std::unique_ptr<std::function<void(sycl::queue*, int)>>(reinterpret_cast<std::function<void(sycl::queue*, int)>*>(userData));
			(*func)(stream, status);
		}
#endif
		// The number of elements this container can store
		size_t m_numel;
		ContainerLocation m_location;

		ContainerStreamType m_associatedStream;
		T* m_buffer;

	};
}

#endif //!__CONTAINER_H__
