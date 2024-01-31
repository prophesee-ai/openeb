/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#ifndef METAVISION_HAL_BUFFER_ALLOCATOR_H
#define METAVISION_HAL_BUFFER_ALLOCATOR_H

#include <memory>
#include <stdlib.h>
#include <limits.h>
#include <system_error>
#include <vector>

namespace Metavision {

namespace BufferAllocatorInternal {
/// @brief a class that manages BufferAllocator<T> allocations, in a type-agnostic way to allow rebinding
///
template<typename T>
class Allocator {
public:
    virtual T *allocate(size_t n)            = 0;
    virtual void deallocate(T *p, size_t n)  = 0;
    virtual size_t max_size() const noexcept = 0;
    virtual ~Allocator() {}
};

template<typename T>
class VectorAllocator : public Allocator<T> {
    using AllocatorType = typename std::vector<T>::allocator_type;
    AllocatorType allocator;

public:
    virtual T *allocate(size_t n) override {
        return allocator.allocate(n);
    }
    virtual void deallocate(T *p, size_t n) override {
        return allocator.deallocate(p, n);
    }
    virtual size_t max_size() const noexcept override {
        return allocator.max_size();
    }
    virtual ~VectorAllocator() {}
};
} // namespace BufferAllocatorInternal

/// @brief An allocator meant for the DataTransfer Buffers
///
/// The base behavior is the one from std::allocator, but it is written so that it may be overriden, in first intent to
/// map driver-allocated buffers into the std::vector used as DataTransfer buffers, keeping the benefit of iterators
/// and other commodities offered by vector
template<typename Data>
class BufferAllocator {
    template<typename T>
    friend bool operator==(const BufferAllocator<T> &lhs, const BufferAllocator<T> &rhs) noexcept;

    template<typename T>
    friend void swap(BufferAllocator<T> &a, BufferAllocator<T> &b);

public:
    using pointer            = Data *;
    using const_pointer      = const Data *;
    using void_pointer       = void *;
    using const_void_pointer = const void *;
    using value_type         = Data;
    using size_type          = size_t;
    using difference_type    = ptrdiff_t;
    // If a datatransfer uses special memory for buffers, the copy of such a buffer should be in normal memory, to
    // avoid running out of special memory. Copies are not expected to go back to the BufferPool
    using propagate_on_container_copy_assignment = std::false_type;
    // On the opposite, if the allocated memory is moved to a different container, the allocator shall be moved as well
    // so that the memory can be properly freed when the buffer is destroyed
    using propagate_on_container_move_assignment = std::true_type;
    // And swapping is the same as moving
    using propagate_on_container_swap = std::true_type;
    using is_always_equal             = std::false_type;

public:
    // Default constuctor, default construction is fine
    BufferAllocator() : impl_(std::make_shared<DefaultAllocator>()) {}
    // Copy constuctor, called when a vector is built with an Allocator
    BufferAllocator(const BufferAllocator &orig) : impl_(orig.impl_) {}
    // Move constructor, move the internal state
    BufferAllocator(BufferAllocator &&orig) : impl_(std::move(orig.impl_)) {}

    template<typename U>
    struct rebind {
        using other = BufferAllocator<U>;
    };
    template<typename U>
    BufferAllocator(const BufferAllocator<U> &orig) : BufferAllocator(orig.get_impl()) {}

public:
    pointer allocate(size_type n) {
        return reinterpret_cast<pointer>(impl_->allocate(n));
    }

    void deallocate(pointer p, size_type n) {
        impl_->deallocate(p, n);
    }

    size_type max_size() const noexcept {
        return impl_->max_size();
    }

    template<class U, class... Args>
    void construct(U *p) {
        // This redefines default construction of the buffer elements, but keeps the normal call to the constructor
        // when there are arguments (such as an init value), via void construct(U *p, Args &&...args);
        // Without this, the vector needs to be resized to max transfer size before transfering new data, needlessly
        // writing default value in the buffer before overwriting it with the actual transfer. This may be costly
        // on CPU time and cache eviction.
    }

public:
    using Impl             = BufferAllocatorInternal::Allocator<Data>;
    using DefaultAllocator = BufferAllocatorInternal::VectorAllocator<Data>;

    // The goal of this design is to manage Buffer Allocation from reserved memory (e.g. contiguous memory).
    // The implementation could be a unique_ptr to a state containing a reference or a shared pointer to the
    // memory pool, plus information regarding the current allocation
    // however, the std::vector Allocator is Copy-constructed and not Move-constructed, which means it is mandatory to
    // handle the copy, preserving the derived type, but using a different allocation (eg a different buffer index in
    // the pool.
    // Having a single Impl referenced several times through shared pointer avoids a layer of objects, and
    // makes copy easier to manage, but requires the shared implementation to keep a map of the allocations.
    using ImplPtr = std::shared_ptr<Impl>;

    // To allow derivated classes to set their state
    BufferAllocator(const ImplPtr &impl) : impl_(impl) {}
    BufferAllocator(ImplPtr &&impl) : impl_(std::move(impl)) {}
    // To allow derivated DataTransfer to break abstraction (via rtti) and get allocation information
    const ImplPtr &get_impl() const {
        return impl_;
    }

private:
    ImplPtr impl_;
};

template<typename Data>
bool operator==(const BufferAllocator<Data> &lhs, const BufferAllocator<Data> &rhs) noexcept {
    return lhs.impl_.get() == rhs.impl_.get();
}

template<typename Data>
bool operator!=(const BufferAllocator<Data> &lhs, const BufferAllocator<Data> &rhs) noexcept {
    return !(lhs == rhs);
}

template<typename Data>
void swap(BufferAllocator<Data> &a, BufferAllocator<Data> &b) {
    a.impl_.swap(b.impl_);
}

} // namespace Metavision

#endif // METAVISION_HAL_BUFFER_ALLOCATOR_H
