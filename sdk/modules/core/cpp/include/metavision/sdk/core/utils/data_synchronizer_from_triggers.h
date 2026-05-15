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

#ifndef METAVISION_SDK_CORE_DATA_SYNCHRONIZER_FROM_TRIGGERS_H
#define METAVISION_SDK_CORE_DATA_SYNCHRONIZER_FROM_TRIGGERS_H

#include <stdexcept>
#include <deque>
#include <condition_variable>
#include <cmath>
#include <functional>

#include "metavision/sdk/base/events/event_ext_trigger.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/core/utils/detail/iterator_traits.h"

namespace Metavision {

/// @brief The purpose of this class is to synchronize with the event-stream the data from an external source generating
/// external triggers
///
/// @warning This class considers that the input stream of triggers is periodic and that the period doesn't change at
/// runtime.
/// @warning It is assumed here that the source of external triggers is the same as the external data's one
///
/// Synchronization information is generated from the input triggers events. This information contains a
/// timestamp (the one of the trigger used to generate it) and an index. The indexes are guaranteed to be strictly and
/// incrementally increasing from a starting value (0 by default). However, some triggers may be lost for various
/// reasons, as a consequence the indexes are generated using both input triggers' timestamp and the expected period
/// of the external data:
///     -> index_increment = (trigger_ts - previous_trigger_ts)/period.
///
/// The synchronization is done by matching indexes generated from the triggers with the ones provided by the
/// external data streams. When a data's index finds a match, its timestamp takes the value of its match.
///
/// This class interpolates synchronization information if triggers are missing (i.e. index_increment > 1). This is to
/// guarantee that all data to synchronize can find a match.
///     -> interpolated_ts = previous_trigger_ts + period
///
/// Only one polarity is used for the synchronization (i.e. up or down, chosen by the user) as it is
/// considered that each external data generates a pair of triggers (i.e. one data for two triggers).
///
/// The synchronization routines are blocking and thread safe.
///
class DataSynchronizerFromTriggers {
public:
    /// @brief Synchronization information data used to timestamp a piece of external data using an external trigger and
    /// the index of the piece of external data
    struct SynchronizationInformation {
        /// Timestamp of the trigger used to generate this
        timestamp t;

        /// Index computed from the triggers timestamps
        uint32_t index;
    };

    /// @brief Parameters to be used to configure the @ref DataSynchronizerFromTriggers
    struct Parameters {
        Parameters(uint32_t period_us);

        double periodicity_tolerance_factor_{0.1}; ///< Used to compute the time interval around the expected trigger's
                                                   ///< timestamp. If the timestamp falls before this range, the trigger
                                                   ///< is considered to be a duplicate and is not used. Though this
                                                   ///< should not happen.
        uint32_t period_us_; ///< Triggers' pair period i.e. expected dt between two triggers of the same polarity.
        uint32_t to_discard_count_{0}; ///< Number of triggers to discard from the beginning of the external triggers.
        uint32_t index_offset_{0};     ///< This is the very first data to synchronize expected index and so the first
                                       ///< trigger index.
        bool reference_polarity_{0};   ///< The trigger's polarity to use.
    };

public:
    /// @brief Constructor
    /// @param parameters The @ref Parameters to use to configure this object.
    DataSynchronizerFromTriggers(const Parameters &parameters);

    /// @brief Destructor
    /// Sets the triggers source as done to unlock the synchronization if it is waiting.
    ~DataSynchronizerFromTriggers();

    /// @brief Resets the synchronization states variables.
    /// Unlocks any pending synchronization before clearing the synchronization information remaining to be used.
    void reset_synchronization();

    /// @brief Notifies this object that the synchronization is done
    ///
    /// This means that either the trigger source and thus no trigger is to be received anymore, or that the external
    /// data stream won't use triggers anymore
    ///
    /// This unlocks pending synchronization and forbids indexing of new triggers.
    void set_synchronization_as_done();

    /// @brief Generates @ref SynchronizationInformation from the external triggers input stream
    ///
    /// This information is to be used for the synchronization (@ref synchronize_data_from_triggers).
    /// This method is not blocking: this is not a synchronization point.
    ///
    /// @tparam ExtTriggerIterator The type of the external trigger input events iterator
    /// @param trigger_it The first iterator to an external trigger to process
    /// @param trigger_it_end The last iterator to an external trigger to process
    /// @warning ExtTriggerIterator must be an iterator over EventExtTrigger
    /// @return The number of external triggers that have been updated
    template<typename ExtTriggerIterator>
    size_t index_triggers(ExtTriggerIterator trigger_it, ExtTriggerIterator trigger_it_end);

    /// @brief Generates @ref SynchronizationInformation from the external triggers input stream
    ///
    /// This information is to be used for the synchronization (@ref synchronize_data_from_triggers).
    /// This method is not blocking: this is not a synchronization point.
    ///
    /// @tparam ExtTriggerIterator The type of the external trigger input events iterator
    /// @param trigger_it The first iterator to an external trigger to process
    /// @param trigger_it_end The last iterator to an external trigger to process
    /// @param indexed_trigger_inserter_it An iterator to insert all triggers used to generate synchronization
    /// information
    /// @warning ExtTriggerIterator must be an iterator over EventExtTrigger
    template<typename ExtTriggerIterator, typename IndexTriggerInserterIterator>
    void index_triggers(ExtTriggerIterator trigger_it, ExtTriggerIterator trigger_it_end,
                        IndexTriggerInserterIterator indexed_trigger_inserter_it);

    /// @brief Synchronizes indexed data with the generated @ref SynchronizationInformation from method
    /// @ref index_triggers
    /// @warning It is assumed here that the source of external triggers is the same as the external data source
    /// @warning It is assumed here that the data source to be synchronized has been indexed before the call.
    /// @warning It is assumed here that the data source' indices are strictly increasing otherwise this may break the
    /// synchronization procedure.
    /// @warning Timestamp of input data source must be accessible and modifiable.
    /// @warning Index of input data source must be accessible.
    /// @tparam DataIterator The input data's iterator type
    /// @param data_it The first iterator to the data to synchronize. Each data is accessed and modified.
    /// @param data_it_end The last iterator to the data to synchronize.
    /// @param data_timestamp_accessor A cb to access the timestamp of the input data. Timestamp must be modifiable.
    /// @param data_index_accessor A cb to access the index of the input data. Neither the index nor the data is
    /// modified.
    /// @return The number of synchronized data. This amount is usually the same as std::distance(data_it, data_it_end)
    /// unless we stopped receiving triggers.
    template<typename DataIterator>
    uint32_t synchronize_data_from_triggers(
        DataIterator data_it, DataIterator data_it_end,
        std::function<timestamp &(detail::value_t<DataIterator> &)> data_timestamp_accessor,
        std::function<uint32_t(const detail::value_t<DataIterator> &)> data_index_accessor);

    /// @brief Indicates the maximum number of external triggers that can be pending to be used for synchronization
    ///
    /// This can be used to slow down an event-based data polling thread
    /// @param max_remaining_to_be_consumed The maximum number of pending external triggers
    void wait_for_triggers_consumed(uint32_t max_remaining_to_be_consumed = 0);

private:
    /// Queue of synchronization information generated from received external triggers
    std::deque<SynchronizationInformation> synchronization_information_deque_;

    /// Parameters
    Parameters parameters_;

    /// States if at least one trigger has been indexed
    bool first_trigger_indexed_;

    /// Sets this triggers source as done i.e. we don't expect to receive anymore events
    bool triggers_source_is_done_;

    /// State variable to keep the count of the last generated index
    uint32_t last_synchronization_index_;

    /// Last received trigger's timestamp
    timestamp last_synchronization_ts_us_;

    /// Safety variable
    std::mutex triggers_updated_mutex_;
    std::condition_variable wait_for_triggers_cond_;
    std::condition_variable wait_for_triggers_consumed_cond_;
};

} // namespace Metavision

#include "metavision/sdk/core/utils/detail/data_synchronizer_from_triggers_impl.h"

#endif // METAVISION_SDK_CORE_DATA_SYNCHRONIZER_FROM_TRIGGERS_H
