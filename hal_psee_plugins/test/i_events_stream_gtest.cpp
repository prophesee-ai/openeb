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

#include <thread>
#include <condition_variable>

#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/hal/facilities/i_event_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/utils/data_transfer.h"
#include "metavision/hal/utils/file_data_transfer.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/utils/gtest/gtest_custom.h"
#include "boards/rawfile/psee_raw_file_header.h"
#include "geometries/vga_geometry.h"
#include "decoders/evt2/evt2_decoder.h"
#include "metavision/hal/facilities/i_decoder.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "tencoder.h"
#include "gen3CD_device.h"
#include "device_builder_maker.h"

using namespace Metavision;

class MockHWIdentification : public I_HW_Identification {
public:
    MockHWIdentification(const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info, long system_id) :
        I_HW_Identification(plugin_sw_info), system_id_(system_id) {}

    virtual std::string get_serial() const override {
        return dummy_serial_;
    }
    virtual long get_system_id() const override {
        return system_id_;
    }
    virtual SensorInfo get_sensor_info() const override {
        return SensorInfo();
    }
    virtual long get_system_version() const override {
        return dummy_system_version_;
    }
    virtual std::vector<std::string> get_available_raw_format() const override {
        std::vector<std::string> available_formats;
        available_formats.push_back(std::string("EVT2"));
        return available_formats;
    }
    virtual std::string get_integrator() const override {
        return dummy_integrator_name_;
    }
    virtual SystemInfo get_system_info() const override {
        return SystemInfo();
    }
    virtual std::string get_connection_type() const override {
        return std::string();
    }

    virtual RawFileHeader get_header_impl() const override {
        PseeRawFileHeader header(*this, VGAGeometry());
        header.set_sub_system_id(dummy_sub_system_id_);
        header.set_format(is_evt3 ? "EVT3" : "EVT2");
        header.set_field(dummy_custom_key_, dummy_custom_value_);
        return header;
    }

    long system_id_;
    static const std::string dummy_serial_;
    static const std::string dummy_integrator_name_;
    static const std::string dummy_custom_key_;
    static const std::string dummy_custom_value_;

    static constexpr bool is_evt3               = false;
    static constexpr long dummy_sub_system_id_  = 3;
    static constexpr long dummy_system_version_ = 0;
};

constexpr long MockHWIdentification::dummy_system_version_;
constexpr long MockHWIdentification::dummy_sub_system_id_;
constexpr bool MockHWIdentification::is_evt3;
const std::string MockHWIdentification::dummy_serial_          = "dummy_serial";
const std::string MockHWIdentification::dummy_integrator_name_ = "integator_name";
const std::string MockHWIdentification::dummy_custom_key_      = "custom";
const std::string MockHWIdentification::dummy_custom_value_    = "field";

class MockDataTransfer : public DataTransfer {
public:
    MockDataTransfer() : DataTransfer(4) {}

    void trigger_transfer(const std::vector<Data> &data) {
        buffer_->clear();
        buffer_->insert(buffer_->end(), data.cbegin(), data.cend());
        transfer_data(buffer_);
    }

private:
    void start_impl(BufferPtr buffer) final {
        buffer_ = buffer;
    }

    void run_impl() final {
        while (!should_stop()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    BufferPtr buffer_;
};

class I_EventsStream_GTest : public GTestWithTmpDir {
public:
    I_EventsStream_GTest(long system_id = metavision_device_traits<Gen3CDDevice>::SYSTEM_ID_DEFAULT) :
        system_id_(system_id) {
        reset();
    }

    virtual ~I_EventsStream_GTest() {}

    void reset() {
        dt_ = new MockDataTransfer();
        std::unique_ptr<MockDataTransfer> dt(dt_);
        // needed by MockHWIdentification::get_header
        auto plugin_sw_info =
            std::make_shared<I_PluginSoftwareInfo>(dummy_plugin_name_, SoftwareInfo(0, 0, 0, "", "", "", ""));
        // needed by MockEventsStream::log_raw_data
        hw_identification_ = std::make_shared<MockHWIdentification>(plugin_sw_info, system_id_);
        events_stream_     = std::make_shared<I_EventsStream>(std::move(dt), hw_identification_);
        events_stream_->start();
    }

    static const std::string dummy_plugin_name_;

protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}

    // Needed facilities,
    std::unique_ptr<Device> device_;
    std::shared_ptr<MockHWIdentification> hw_identification_;
    std::shared_ptr<I_EventsStream> events_stream_;
    MockDataTransfer *dt_ = nullptr;
    long system_id_;
};

const std::string I_EventsStream_GTest::dummy_plugin_name_ = "plugin_name";

TEST_F(I_EventsStream_GTest, add_sub_system_id_to_header) {
    // Create tmp file
    std::string filename = tmpdir_handler_->get_full_path("log.raw");
    events_stream_->log_raw_data(filename);
    events_stream_->stop_log_raw_data();

    // Now read the subid from the header and verify that it is the one set
    std::ifstream file(filename);
    auto header = PseeRawFileHeader(file);
    ASSERT_EQ(MockHWIdentification::dummy_sub_system_id_, header.get_sub_system_id());
    file.close();
}

TEST_F(I_EventsStream_GTest, poll_buffer) {
    ASSERT_EQ(0, events_stream_->poll_buffer());
    std::vector<DataTransfer::Data> data = {1, 2, 3, 4, 5};
    dt_->trigger_transfer(data);
    ASSERT_EQ(1, events_stream_->poll_buffer());
}

TEST_F(I_EventsStream_GTest, add_data_triggers_wait_next_buffer) {
    std::condition_variable wait_var;
    std::mutex triggered_mutex;
    bool triggered = false;
    auto thread    = std::thread([this, &wait_var, &triggered_mutex, &triggered]() {
        EXPECT_EQ(1, events_stream_->wait_next_buffer()); // waits for a buffer
        {
            std::unique_lock<std::mutex> wait_lock(triggered_mutex);
            triggered = true;
        }
        wait_var.notify_one();
    });

    while (!thread.joinable()) {} // wait thread to be started

    // trigger with add data
    std::vector<DataTransfer::Data> data = {1, 2, 3, 4, 5};
    dt_->trigger_transfer(data);

    // Wait for the trigger to be processed (add a timeout to not wait forever)
    auto now = std::chrono::system_clock::now();
    {
        std::unique_lock<std::mutex> wait_lock(triggered_mutex);
        wait_var.wait_until(wait_lock, now + std::chrono::seconds(1), [&triggered]() { return triggered; });
    }

    EXPECT_TRUE(
        triggered); // check that we did trigger the condition and that we did not wake up because of the timeout
    events_stream_->stop();
    thread.join();
}

TEST_F(I_EventsStream_GTest, stop_triggers_wait_next_buffer) {
    std::condition_variable wait_var;
    std::mutex triggered_mutex_;
    bool triggered = false;
    auto thread    = std::thread([this, &wait_var, &triggered_mutex_, &triggered]() {
        events_stream_->wait_next_buffer(); // waits for a trigger
        {
            std::unique_lock<std::mutex> wait_lock(triggered_mutex_);
            triggered = true;
        }
        wait_var.notify_one();
    });

    while (!thread.joinable()) {} // wait thread to be started

    // trigger with stop
    events_stream_->stop();

    // Wait for the trigger to be processed (add a timeout to not wait forever)
    auto now = std::chrono::system_clock::now();
    std::unique_lock<std::mutex> wait_lock(triggered_mutex_);
    wait_var.wait_until(wait_lock, now + std::chrono::seconds(1),
                        [&triggered]() { return triggered; }); // to avoid dead lock

    EXPECT_TRUE(
        triggered); // check that we did trigger the condition and that we did not wake up because of the timeout
    events_stream_->stop();
    thread.join();
}

TEST_WITH_DATASET(EventsStream_GTest, stop_on_recording_does_not_drop_buffers) {
    // Read the dataset provided
    std::string dataset_file_path =
        (boost::filesystem::path(GtestsParameters::instance().dataset_dir) / "openeb" / "gen4_evt3_hand.raw").string();

    std::unique_ptr<Metavision::Device> device = Metavision::DeviceDiscovery::open_raw_file(dataset_file_path);
    EXPECT_TRUE(device != nullptr);

    long int n_raw                             = 0;
    Metavision::I_EventsStream *i_eventsstream = device->get_facility<Metavision::I_EventsStream>();
    while (true) {
        i_eventsstream->start();
        // allow reading thread to accumulate some buffers that could be dropped when we stop the events stream
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (i_eventsstream->wait_next_buffer() < 0) {
            break;
        }
        long int n_rawbytes                = 0;
        I_EventsStream::RawData *ev_buffer = i_eventsstream->get_latest_raw_data(n_rawbytes);
        n_raw += n_rawbytes;
        i_eventsstream->stop();
    }

    ASSERT_EQ(96786804, n_raw);
}

template<class Device>
class I_EventsStreamT_GTest : public I_EventsStream_GTest {
public:
    I_EventsStreamT_GTest() : I_EventsStream_GTest(metavision_device_traits<Device>::SYSTEM_ID_DEFAULT) {
        build_events();
    }

    virtual ~I_EventsStreamT_GTest() {}

protected:
    void build_events();

    I_Decoder *create_decoder();

    virtual void SetUp() override {
        I_EventsStream_GTest::SetUp();
    }

    virtual void TearDown() override {
        I_EventsStream_GTest::TearDown();
    }

    std::vector<EventCD> events1_, events2_;
};

template<>
I_Decoder *I_EventsStreamT_GTest<Gen3CDDevice>::create_decoder() {
    static constexpr bool TimeShiftingEnabled = false;

    DeviceBuilder device_builder = make_device_builder();

    auto cd_event_decoder          = device_builder.add_facility(std::make_unique<I_EventDecoder<EventCD>>());
    auto ext_trigger_event_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventExtTrigger>>());
    auto decoder                   = device_builder.add_facility(
        std::make_unique<EVT2Decoder>(TimeShiftingEnabled, cd_event_decoder, ext_trigger_event_decoder));
    device_ = device_builder();

    return decoder.get();
}

template<>
void I_EventsStreamT_GTest<Gen3CDDevice>::build_events() {
    events1_ = {
        {16, 345, 1, 1642},   {3, 360, 0, 3292},    {365, 61, 1, 4977},   {119, 44, 0, 6631},   {24, 349, 1, 8258},
        {41, 339, 1, 9908},   {132, 52, 0, 11577},  {373, 106, 1, 13210}, {2, 334, 1, 14842},   {8, 379, 1, 16516},
        {329, 89, 1, 18179},  {45, 380, 1, 19810},  {22, 350, 0, 21510},  {329, 93, 1, 23207},  {67, 249, 0, 24944},
        {11, 240, 1, 26619},  {12, 353, 1, 28315},  {35, 422, 1, 30047},  {256, 117, 1, 31821}, {14, 311, 0, 33520},
        {128, 66, 1, 35253},  {44, 284, 0, 37005},  {72, 248, 0, 38783},  {9, 369, 1, 40490},   {30, 323, 1, 42220},
        {16, 325, 1, 43976},  {45, 321, 1, 45754},  {26, 322, 1, 47477},  {30, 317, 1, 49236},  {116, 57, 1, 50977},
        {249, 47, 1, 52740},  {122, 47, 0, 54440},  {12, 373, 1, 56164},  {43, 332, 0, 57866},  {38, 369, 1, 59574},
        {2, 328, 0, 61313},   {16, 355, 1, 63035},  {32, 308, 1, 64797},  {296, 109, 0, 66553}, {40, 308, 1, 68284},
        {358, 54, 1, 70002},  {287, 47, 1, 71769},  {326, 28, 1, 73513},  {68, 223, 0, 75246},  {34, 304, 1, 77010},
        {359, 53, 1, 78756},  {72, 222, 0, 80496},  {1, 326, 0, 82262},   {284, 105, 0, 84020}, {123, 36, 1, 85803},
        {142, 54, 1, 87545},  {12, 278, 0, 89275},  {287, 112, 1, 91028}, {278, 102, 0, 92819}, {11, 334, 1, 94596},
        {40, 287, 1, 96320},  {120, 32, 1, 98113},  {78, 223, 0, 99845},  {53, 336, 1, 101530}, {345, 47, 0, 103286},
        {35, 295, 0, 105032}, {8, 321, 1, 106761},  {254, 36, 1, 108485}, {23, 281, 1, 110214}, {36, 296, 0, 111941},
        {4, 279, 1, 113679},  {239, 51, 1, 115390}, {50, 251, 0, 117156}};
}

typedef ::testing::Types<Gen3CDDevice> TestingTypes;

TYPED_TEST_CASE(I_EventsStreamT_GTest, TestingTypes);

TYPED_TEST(I_EventsStreamT_GTest, test_log) {
    // Get tmp file name
    std::string filename(this->tmpdir_handler_->get_full_path("log.raw"));

    // Firmware, serial, system_id and data are added through the
    // common events stream automatically
    this->events_stream_->log_raw_data(filename);

    using EvtFormat = typename metavision_device_traits<TypeParam>::RawEventFormat;

    TEncoder<EvtFormat> encoder;
    std::vector<DataTransfer::Data> data;

    encoder.set_encode_event_callback(
        [&data](const uint8_t *ev, const uint8_t *ev_end) { data.insert(data.end(), ev, ev_end); });

    // Encode
    encoder.encode(this->events1_.data(), this->events1_.data() + this->events1_.size());
    encoder.flush();
    long n_events;
    this->events_stream_->get_latest_raw_data(n_events);
    this->dt_->trigger_transfer(data);

    // REMARK : as of today, in order to log we have to call
    // get_latest_raw_data before add_data and after
    // TODO : remove this following line if we modify the behaviour
    // of log_data (for example if we log when calling add_data or if
    // we log in a separate thread)
    this->events_stream_->get_latest_raw_data(n_events);
    this->events_stream_->stop_log_raw_data();

    // Now open the file and verify what is written :
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    ASSERT_TRUE(file.is_open());

    // Get the header
    PseeRawFileHeader header_read(file);
    ASSERT_FALSE(header_read.empty());
    ASSERT_EQ(MockHWIdentification::dummy_serial_, header_read.get_serial());
    ASSERT_EQ(MockHWIdentification::is_evt3 ? "EVT3" : "EVT2", header_read.get_format());
    ASSERT_EQ(this->system_id_, header_read.get_system_id());
    ASSERT_EQ(MockHWIdentification::dummy_sub_system_id_, header_read.get_sub_system_id());
    ASSERT_EQ(MockHWIdentification::dummy_integrator_name_, header_read.get_integrator_name());
    ASSERT_EQ(MockHWIdentification::dummy_custom_value_,
              header_read.get_field(MockHWIdentification::dummy_custom_key_));

    // Read the file

    // Check length
    auto position_first_data = file.tellg();                // Get the current position in the file
    file.seekg(0, std::ios::end);                           // Go to the end of the file
    auto n_tot_data = (file.tellg() - position_first_data); // Compute the number of data
    file.seekg(position_first_data);                        // Reset the position at the
                                                            // first event
    char *buffer = new char[n_tot_data];

    // Read data as a block:
    file.read(buffer, n_tot_data);
    ASSERT_EQ(n_tot_data, file.gcount());

    // Verify we have read everything
    file.read(buffer, n_tot_data);
    ASSERT_EQ(0, file.gcount());
    ASSERT_TRUE(file.eof());
    file.close();

    // Decode the buffer received :
    uint8_t *buffer_to_decode = reinterpret_cast<uint8_t *>(buffer);

    // Decoder
    auto decoder = this->create_decoder();

    std::vector<EventCD> decoded_events;
    auto td_decoder = this->device_->template get_facility<I_EventDecoder<EventCD>>();
    td_decoder->add_event_buffer_callback([&decoded_events](const EventCD *begin, const EventCD *end) {
        decoded_events.insert(decoded_events.end(), begin, end);
    });

    decoder->decode(buffer_to_decode, buffer_to_decode + n_tot_data);

    ASSERT_EQ(this->events1_.size(), decoded_events.size());
    auto it_expected = this->events1_.begin();
    auto it          = decoded_events.begin();

    using SizeType = std::vector<EventCD>::size_type;
    for (SizeType i = 0, max_i = this->events1_.size(); i < max_i; ++i, ++it, ++it_expected) {
        EXPECT_EQ(it_expected->x, it->x);
        EXPECT_EQ(it_expected->y, it->y);
        EXPECT_EQ(it_expected->p, it->p);
        EXPECT_EQ(it_expected->t, it->t);
    }

    delete[] buffer;
}
