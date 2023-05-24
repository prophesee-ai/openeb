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

#ifndef METAVISION_SDK_BASE_OBJECT_POOL_H
#define METAVISION_SDK_BASE_OBJECT_POOL_H

#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <stack>
#include <stdexcept>
#include <type_traits>

namespace Metavision {

/// @brief Class that creates a reusable pool of heap allocated objects
///
/// The @a ObjectPool allocates objects that are returned to the pool upon destruction.
/// @ref acquire is used to allocate or re-use a previously allocated object instance.
/// The object are made available through a @a std::unique_ptr or a @a std::shared_ptr
/// according to the template @a acquire_shared_ptr.
/// The smart pointers are given a custom deleter, which automatically adds the object
/// back to the pool upon destruction.
///
/// @tparam T the type of object stored
/// @tparam acquire_shared_ptr if true, the object are wrapped by a @a std::shared_ptr, otherwise
/// a std::unique_ptr is returned instead
template<class T, bool acquire_shared_ptr = false>
class ObjectPool {
private:
    struct Impl;
    struct Deleter {
        explicit Deleter(std::weak_ptr<Impl> pool) : pool_(pool) {}
        void operator()(T *ptr) {
            if (auto pool_ptr = pool_.lock())
                try {
                    pool_ptr->add(std::unique_ptr<T>{ptr});
                } catch (...) {
                    // Out of memory. Free some
                    std::default_delete<T>{}(ptr);
                }
            else
                std::default_delete<T>{}(ptr);
        }

    private:
        std::weak_ptr<Impl> pool_;
    };

public:
    using ptr_type =
        typename std::conditional<acquire_shared_ptr, std::shared_ptr<T>, std::unique_ptr<T, Deleter>>::type;

    /// @brief Creates an object pool with limited number of objects that can be allocated
    ///
    /// There won't be memory allocation upon call to @ref acquire if all objects in the memory pool are already
    /// used.
    ///
    /// @param num_initial_objects Number of objects initially allocated in the pool
    /// @return An object pool with bounded memory
    static ObjectPool<T, acquire_shared_ptr> make_bounded(size_t num_initial_objects = 64) {
        return ObjectPool(num_initial_objects, true);
    }

    /// @brief Creates an object pool with limited number of objects that can be allocated
    ///
    /// There won't be memory allocation upon call to @ref acquire if all objects in the memory pool are already
    /// used.
    ///
    /// @param num_initial_objects Number of objects initially allocated in the pool
    /// @param args The arguments forwarded to the object constructor during allocation
    /// @return An object pool with bounded memory
    template<typename... Args>
    static ObjectPool<T, acquire_shared_ptr> make_bounded(size_t num_initial_objects, Args &&...args) {
        return ObjectPool(num_initial_objects, true, std::forward<Args>(args)...);
    }

    /// @brief Creates an object pool with expendable memory usage
    ///
    /// A pool with unbounded memory will allocate a new object when all objects in the pool are already used
    /// and @ref acquire is called.
    ///
    /// @param num_initial_objects Number of objects initially allocated in the pool
    /// @return An object pool with unbounded memory
    template<typename... Args>
    static ObjectPool<T, acquire_shared_ptr> make_unbounded(size_t num_initial_objects = 64) {
        return ObjectPool(num_initial_objects, false);
    }

    /// @brief Creates an object pool with expendable memory usage
    ///
    /// A pool with unbounded memory will allocate a new object when all objects in the pool are already used
    /// and @ref acquire is called.
    ///
    /// @param num_initial_objects Number of objects initially allocated in the pool
    /// @param args The arguments forwarded to the object constructor during allocation
    /// @return An object pool with unbounded memory
    template<typename... Args>
    static ObjectPool<T, acquire_shared_ptr> make_unbounded(size_t num_initial_objects, Args &&...args) {
        return ObjectPool(num_initial_objects, false, std::forward<Args>(args)...);
    }

    /// @brief Default constructor that builds an unbounded object pool with an initial number of objects
    /// allocated
    /// @ref make_unbounded
    ObjectPool() : ObjectPool(64, false) {
        static_assert(std::is_default_constructible<T>::value, "Using ObjectPool default constructor: object "
                                                               "must be default constructible. Otherwise, use static "
                                                               "build method.");
    };

    /// @brief Adds an object to the pool
    /// @param t A unique_ptr storing the object
    void add(std::unique_ptr<T> t) {
        impl_->add(std::move(t));
    }

    /// @brief Allocates or re-use a previously allocated object
    /// @param args Optional arguments to be passed when allocating the object
    /// @return A unique or shared pointer to the allocated object
    template<typename... Args>
    ptr_type acquire(Args &&...args) {
        return impl_->acquire(std::forward<Args>(args)...);
    }

    /// @brief Checks if the pool is empty
    /// @return true if the pool is empty, false if the pool contains object ready to be re-used
    bool empty() const {
        return impl_->empty();
    }

    /// @brief Gets the number of objects in the pool
    /// @return The number of previously allocated and ready to-reuse objects in the pool
    size_t size() const {
        return impl_->size();
    }

    /// @brief Checks the memory pool type i.e. bounded or unbounded
    /// @return true if the memory pool is bounded, false if it is unbounded
    bool is_bounded() const {
        return impl_->is_bounded();
    }

    /// @brief Ensure that the pool contains 'size' available objects, ready to be acquired.
    /// @note Only works for unbounded pool
    /// @param size The maximum number of objects to be available
    /// @param args Optional arguments to be used when allocating the object
    /// @return the number of newly allocated object in the pool
    template<typename... Args>
    size_t arrange(size_t size, Args &&...args) {
        return impl_->arrange(size, std::forward<Args>(args)...);
    }

private:
    /// @brief Constructor
    template<typename... Args>
    ObjectPool(size_t num_initial_objects, bool bounded_memory, Args &&...args) :
        impl_(new Impl(num_initial_objects, bounded_memory, std::forward<Args>(args)...)) {}

    /// @brief Implementation of the object pool in a separate object
    ///
    /// This is defined to make movable and move assignable the object pool.
    struct Impl : public std::enable_shared_from_this<Impl> {
        /// @brief Constructor
        template<typename... Args>
        Impl(size_t num_initial_objects, bool bounded_memory, Args &&...args) : bounded_memory_(bounded_memory) {
            if (num_initial_objects == 0 && bounded_memory) {
                throw std::invalid_argument(
                    "Failed to allocate memory for the bounded object pool: pool's size can not be 0.");
            }
            for (size_t i = 0; i < num_initial_objects; ++i) {
                pool_.push(std::unique_ptr<T>(new T(std::forward<Args>(args)...)));
            }
        }

        /// @brief Adds an object to the pool
        /// @param t A unique_ptr storing the object
        void add(std::unique_ptr<T> t) {
            std::lock_guard<std::mutex> lock(mutex_);
            pool_.push(std::move(t));
            if (bounded_memory_) {
                cond_.notify_all();
            }
        }

        /// @brief Increase pool capacity to the new size if larger than the actual pool size.
        /// @param size The new pool capacity size
        /// @param args Optional arguments to be used when allocating the object
        /// @return the number of newly allocated object in the pool
        template<typename... Args>
        size_t arrange(size_t size, Args &&...args) {
            if (bounded_memory_ || size <= pool_.size()) {
                return 0;
            }

            std::unique_lock<std::mutex> lock(mutex_);
            size_t nb_allocated_obj = size - pool_.size();
            while (pool_.size() < size) {
                pool_.push(std::unique_ptr<T>(new T(std::forward<Args>(args)...)));
            }

            return nb_allocated_obj;
        }

        /// @brief Allocates or re-use a previously allocated object
        /// @param args Optional arguments to be passed when allocating the object
        /// @return A unique or shared pointer to the allocated object
        template<typename... Args>
        ptr_type acquire(Args &&...args) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (pool_.empty()) {
                if (bounded_memory_) {
                    cond_.wait(lock, [this] { return !pool_.empty(); });
                } else {
                    pool_.push(std::unique_ptr<T>(new T(std::forward<Args>(args)...)));
                }
            }
            ptr_type tmp(pool_.top().release(), Deleter{this->shared_from_this()});
            pool_.pop();
            return tmp;
        }

        /// @brief Checks if the pool is empty
        /// @return true if the pool is empty, false if the pool contains an object ready to be re-used
        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return pool_.empty();
        }

        /// @brief Gets the number of objects in the pool
        /// @return The number of previously allocated and ready to-reuse objects in the pool
        size_t size() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return pool_.size();
        }

        /// @brief Checks the memory pool type i.e. bounded or unbounded
        /// @return true if the memory pool is bounded, false if it is unbounded
        bool is_bounded() const {
            return bounded_memory_;
        }

        mutable std::mutex mutex_;
        mutable std::condition_variable cond_;
        std::stack<std::unique_ptr<T>> pool_;
        bool bounded_memory_{false};
    };

    std::shared_ptr<Impl> impl_;
};

/// @brief Convenience alias to use a @ref ObjectPool returning shared pointers
/// @tparam T the type of object stored in the pool
template<typename T>
using SharedObjectPool = ObjectPool<T, true>;

} // namespace Metavision

#endif // METAVISION_SDK_BASE_OBJECT_POOL_H
