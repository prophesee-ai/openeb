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

#include <thread>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <functional>

#include "metavision/sdk/base/utils/object_pool.h"

namespace Metavision {

/// @brief An interface that manages the data transfer from a source to high level user space
///
/// This object behaves as master: the client of this class is notified when new buffers are transferred or the data
/// transfer is stopped.
/// @warning This class is not thread safe and is used in conjunction with the @ref I_EventsStream or at the user's
/// risk
class DataTransfer {
public:
    /// Alias for the type of the data transferred
    using Data = uint8_t;

    /// Alias for the type of the internal buffer of data
    using Buffer = std::vector<Data>;

    /// Alias for the object handling the buffers pool
    using BufferPool = SharedObjectPool<Buffer>;

    /// Alias for the ptr type returned by the buffer pool
    using BufferPtr = BufferPool::ptr_type;

    /// Alias for a callback called when the data transfer starts or stops transferring data
    enum class Status { Started = 0, Stopped = 1 };
    using StatusChangeCallback_t = std::function<void(Status)>;

    /// Alias for a callback to process transferred buffer of data
    using NewBufferCallback_t = std::function<void(const BufferPtr &)>;

    /// @brief Builds a DataTransfer object
    /// @param raw_event_size_bytes The size of a RAW event in bytes
    DataTransfer(uint32_t raw_event_size_bytes);

    /// @brief Builds a DataTransfer object
    /// @param raw_event_size_bytes The size of a RAW event in bytes
    /// @param buffer_pool A user defined buffer pool to use instead of the default one (unbounded, @ref ObjectPool)
    DataTransfer(uint32_t raw_event_size_bytes, const BufferPool &buffer_pool);

    /// @brief Destructor
    ///
    /// Stops all transfers and wait for all thread to join
    virtual ~DataTransfer();

    /// @brief Starts the transfers
    void start();

    /// @brief Stops the transfers
    void stop();

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
    /// @param cb The cb to call when a new buffer is transfered
    /// @return The id of the callback. This id is unique.
    size_t add_new_buffer_callback(NewBufferCallback_t cb);

    /// @brief Removes the callback with input id
    /// @param cb_id The id of the callback to remove
    /// @note This method is not thread safe. You should add/remove the various callback before starting the transfers
    void remove_callback(size_t cb_id);

protected:
    /// @brief Returns the size of a RAW event in bytes
    ///
    /// This is to help ensuring the integrity of the transferred buffer
    uint32_t get_raw_event_size_bytes() const;

    /// @brief The implementation must call this method whenever a buffer of data is ready to be transferred
    ///
    /// @warning The transferred buffer must hold coherent and continuous data.
    /// @warning The buffer size may not be a multiple of a RAW event byte size if the last event is split, in which
    /// case, the next buffer must contain the remaining bytes of the split event.
    /// @warning The implementation must resize the buffer to the actual size of the transferred data
    /// @param buffer The buffer filled with RAW data to transfer.
    /// @return A buffer taken from the buffer pool
    BufferPtr transfer_data(const BufferPtr &buffer);

    /// @brief Requests a new buffer from the pool
    /// @return A buffer taken from the object pool
    BufferPtr get_buffer();

    /// @brief Returns if the transfer must stop as request by the base class implementation
    ///
    /// This method can be called safely by the child implementation to know whether the implementation of the run
    /// should return
    bool should_stop();

private:
    /// @brief Start transfer implementation.
    ///
    /// This method is called before running the transfer thread.
    /// It can be used as an initialization steps for the implementation.
    /// It also provides a data buffer taken from the object pool to be used for the transfers.
    ///
    /// @note Additional buffers can be obtained by calling @ref get_buffer if multiple transfers are happening at the
    /// same time
    /// @note When this method is called, should_stop returns false
    /// @param buffer A data buffer taken from the pool
    virtual void start_impl(BufferPtr buffer);

    /// @brief Data transfer implementation
    ///
    /// This method must hold the data polling and transfer logic from a source.
    /// It is run in a thread within the base class.
    /// @warning The implementation must ensure that whenever should_stop returns true, this method returns and cleans
    /// up resources
    virtual void run_impl() = 0;

    /// @brief Stop transfer implementation
    ///
    /// This method is here to notify run_impl that it must stop if a call to should_stop() is not sufficient.
    ///
    /// @warning Resources should not be clean up in this method as run_impl could still try to use them. It is advised
    /// to do so in the scope of the run_impl method to avoid concurrent calls
    virtual void stop_impl();

    std::thread run_transfers_thread_;
    BufferPool buffer_pool_;
    std::unordered_map<uint32_t, StatusChangeCallback_t> status_change_cbs_;
    std::unordered_map<uint32_t, NewBufferCallback_t> new_buffer_cbs_;
    const uint32_t raw_event_size_bytes_;
    std::atomic<bool> stop_{false};
    uint32_t cb_index_{0};
};

} // namespace Metavision

#endif // METAVISION_HAL_DATA_TRANSFER_H
