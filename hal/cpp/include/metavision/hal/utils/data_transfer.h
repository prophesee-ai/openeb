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

#ifndef METAVISION_HAL_DATA_TRANSFER_H
#define METAVISION_HAL_DATA_TRANSFER_H

#include <algorithm>
#include <any>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include "metavision/sdk/base/utils/object_pool.h"

namespace Metavision {

/// @brief An interface that manages the data transfer from a source to high level user space
///
/// This object behaves as master: the client of this class is notified when new buffers are transferred or the data
/// transfer is stopped.
///
/// @warning This class is not thread safe and is used in conjunction with the @ref I_EventsStream or at the user's
/// risk
class DataTransfer {
public:
    /// Alias for the type of the data transferred
    using Data = uint8_t;

    /// Convenience alias for a default object handling the buffers pool
    using DefaultBufferPool = SharedObjectPool<std::vector<Data>>;

    /// Convenience alias to a object type from the default type pool
    using DefaultBufferType = DefaultBufferPool::value_type;

    /// Convenience alias for a default object handling the buffers pool
    using DefaultBufferPtr = DefaultBufferPool::ptr_type;

    /// @brief Interface for a Raw Event Data producer
    /// Derived class are responsible from reading Raw Event buffers from the platform and feed those buffer
    /// to the DataTransfer.
    /// @note Buffers are obtained via a Buffer Pool object owned by the derived object. So that the derived object can
    /// customise the required underlying memory type for the specific platform.
    /// @note DefaultBufferPool can be used as a default buffer pool object type.
    class RawDataProducer : public std::enable_shared_from_this<RawDataProducer> {
    public:
        /// @brief Destructor
        virtual ~RawDataProducer() = default;

        /// @brief Start transfer implementation.
        ///
        /// This method is called before running the transfer thread.
        /// It can be used as an initialization steps for the implementation.
        ///
        /// @note When this method is called, should_stop returns false
        virtual void start_impl() {}

        /// @brief Data transfer implementation
        ///
        /// This method must hold the data polling and transfer logic from a source.
        /// It is run in a thread within the base class.
        ///
        ///  When a buffer is ready to be transferred, the implementation must call DataTransfer::transfer_data with the
        ///  buffer
        ///
        /// @warning The implementation must ensure that whenever should_stop returns true, this method returns and
        /// cleans up resources
        virtual void run_impl(const DataTransfer &) = 0;

        /// @brief Stop transfer implementation
        ///
        /// This method is here to notify run_impl that it must stop if a call to should_stop() is not sufficient.
        ///
        /// @warning Resources should not be clean up in this method as run_impl could still try to use them. It is
        /// advised to do so in the scope of the run_impl method to avoid concurrent calls
        virtual void stop_impl() {}
    };

    /// Alias for a shared pointer to a RawDataProducer
    using RawDataProducerPtr = std::shared_ptr<RawDataProducer>;

    /// @brief Generic Buffer type used by DataTransfer
    ///
    /// BufferPtr is a type erased wrapper around a  buffer from a Buffer Pool.
    /// It is used to hold a buffer of data that can be transferred by the DataTransfer down the pipeline without
    /// having them to know about the underlying buffer implementation.
    class BufferPtr {
    public:
        /// @brief Default raw data base type
        using PtrType = const Data *;

        /// @brief Default constructor
        BufferPtr() = default;

        /// @brief Constructor
        /// @param buffer The buffer to hold
        /// @param data The data pointer of the buffer
        /// @param buffer_size The size of the buffer
        BufferPtr(std::any buffer, PtrType data, std::size_t buffer_size);

        /// @brief Equality operator
        bool operator==(const BufferPtr &other) const;

        /// @brief bool conversion operator
        operator bool() const noexcept;

        /// @brief Accessor to the buffer size
        std::size_t size() const noexcept;

        /// @brief Accessor to the buffer data
        PtrType data() const noexcept;

        /// @brief Iterator semantic to the beginning of the buffer
        PtrType begin() const noexcept;

        /// @brief Iterator semantic to the end of the buffer
        PtrType end() const noexcept;

        /// @brief Iterator semantic to the beginning of a const buffer
        const PtrType cbegin() const noexcept;

        /// @brief Iterator semantic to the end of a const buffer
        const PtrType cend() const noexcept;

        /// @brief Clone the buffer by reallocating the data and copying the content
        BufferPtr clone() const;

        using CloneType       = std::vector<Data>;
        using SharedCloneType = std::shared_ptr<CloneType>;

        /// @brief Cast back internally stored buffer to its specific type
        /// Should throw if the type isn't of a cloned one
        SharedCloneType any_clone_cast() const;

        /// @brief Reset the buffer
        void reset() noexcept;

    private:
        std::any internal_buffer_;
        PtrType buffer_data_     = nullptr;
        std::size_t buffer_size_ = 0;
    };

    /// @brief Convenience function to create a BufferPtr from a buffer that originates a Buffer Pool
    /// @param buffer_ptr The buffer to create a BufferPtr from
    /// @return A BufferPtr holding the buffer
    template<typename T>
    static BufferPtr make_buffer_ptr(const T &buffer_ptr) {
        return {buffer_ptr, reinterpret_cast<Data *>(buffer_ptr->data()), buffer_ptr->size()};
    }

    /// Alias for a callback called when the data transfer starts or stops transferring data
    enum class Status { Started = 0, Stopped = 1 };
    using StatusChangeCallback_t = std::function<void(Status)>;

    /// Alias for a callback to process transferred buffer of data
    using NewBufferCallback_t = std::function<void(const BufferPtr &)>;

    /// Alias for a callback to process errors that happened during transfer
    using TransferErrorCallback_t = std::function<void(std::exception_ptr eptr)>;

    /// @brief Builds a DataTransfer object
    /// @param data_producer_ptr A Pointer to a data producer that will provide raw event data
    explicit DataTransfer(RawDataProducerPtr data_producer_ptr);

    /// @brief Destructor
    ///
    /// Stops all transfers and wait for all thread to join
    ~DataTransfer();

    /// @brief Starts the transfers
    void start();

    /// @brief Stops the transfers
    void stop();

    /// @brief Suspends the transfers
    /// @note This function can be used to temporarily suspend transfers, while keeping the
    /// transfer thread alive. To resume the transfers, call @ref resume.
    void suspend();

    /// @brief Resumes the transfers
    void resume();

    /// @brief Check if the transfers are stopped
    /// @return True if the transfers are stopped, false otherwise
    bool stopped() const;

    /// @brief Adds a callback called when the data transfer starts or stops transferring data
    /// @warning This method is not thread safe. You should add/remove the various callback before starting the
    /// transfers
    /// @warning It's not allowed to add/remove a callback from the callback itself
    /// @param cb The cb to call when the status changes
    /// @return The id of the callback. This id is unique.
    size_t add_status_changed_callback(StatusChangeCallback_t cb);

    /// @brief Adds a callback to process transferred buffer of data
    /// @warning This method is not thread safe. You should add/remove the various callback before starting the
    /// transfers
    /// @warning It's not allowed to add/remove a callback from the callback itself
    /// @param cb The cb to call when a new buffer is transferred
    /// @return The id of the callback. This id is unique.
    size_t add_new_buffer_callback(NewBufferCallback_t cb);

    /// @brief Adds a callback to process errors that happened during transfer
    /// @warning This method is not thread safe. You should add/remove the various callback before starting the
    /// transfers
    /// @warning It's not allowed to add/remove a callback from the callback itself
    /// @param cb The cb to call when a new buffer is transferred
    /// @return The id of the callback. This id is unique.
    size_t add_transfer_error_callback(TransferErrorCallback_t cb);

    /// @brief Removes the callback with input id
    /// @param cb_id The id of the callback to remove
    /// @note This method is not thread safe. You should add/remove the various callback before starting the
    /// transfers
    void remove_callback(size_t cb_id);

    /// @brief Returns the data producer used by the transfer
    RawDataProducerPtr get_data_producer() const {
        return data_producer_ptr_;
    };

    /// @brief Convenience function to transfer a buffer of data coming from an object pool by wrapping if to a
    /// BufferPtr
    template<typename T>
    void transfer_data(const T &buffer) const {
        fire_callbacks(make_buffer_ptr(buffer));
    }

    /// @brief Returns whether the transfer must stop as requested by the base class implementation
    ///
    /// This method can be called safely by the child implementation to know whether the implementation of the run
    /// should return
    bool should_stop() const;

    /// @brief Notify the internal thread that it shall stop
    ///
    /// This method makes a thread-safe update of should_stop() return value
    void notify_stop();

    /// @brief Trigger all registered callbacks
    /// @warning The transferred buffer must hold coherent and continuous data.
    /// @warning The buffer size may not be a multiple of a RAW event byte size if the last event is split, in which
    /// case, the next buffer must contain the remaining bytes of the split event.
    /// @warning The implementation must resize the buffer to the actual size of the transferred data
    /// @param buffer The data to be carried forward
    void fire_callbacks(const BufferPtr &buffer) const;

private:
    std::thread run_transfers_thread_;
    RawDataProducerPtr data_producer_ptr_;

    std::unordered_map<uint32_t, StatusChangeCallback_t> status_change_cbs_;
    std::unordered_map<uint32_t, NewBufferCallback_t> new_buffer_cbs_;
    std::unordered_map<uint32_t, TransferErrorCallback_t> transfer_error_cbs_;
    std::atomic<bool> stop_{false};
    uint32_t cb_index_{0};

    std::mutex suspend_mutex_, running_mutex_;
    std::condition_variable suspend_cond_, running_cond_;
    std::atomic<bool> suspend_{false}, running_{false};
};

} // namespace Metavision

template<>
struct std::hash<Metavision::DataTransfer::BufferPtr> {
    std::size_t operator()(const Metavision::DataTransfer::BufferPtr &buffer) const {
        return std::hash<decltype(buffer.data())>{}(buffer.data());
    }
};

#endif // METAVISION_HAL_DATA_TRANSFER_H
