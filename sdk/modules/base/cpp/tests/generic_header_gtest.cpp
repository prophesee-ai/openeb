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

#include <fstream>
#include <sstream>
#include <gtest/gtest.h>
#include <iomanip>

#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/sdk/base/utils/generic_header.h"

TEST(GenericHeader_GTest, default_constructor) {
    // WHEN a header map is built with default constructor
    Metavision::GenericHeader header;

    // THEN header map is empty
    ASSERT_TRUE(header.empty());
}

TEST(GenericHeader_GTest, set_and_remove_field) {
    // GIVEN an empty header map
    Metavision::GenericHeader header;

    // THEN the header is empty
    ASSERT_TRUE(header.empty());

    // WHEN adding a field
    header.set_field("Key", "Value");

    // THEN we can retrieve the field and its value is the same one that was set
    ASSERT_EQ("Value", header.get_field("Key"));

    // AND THEN the header is no longer empty
    ASSERT_FALSE(header.empty());

    // WHEN adding other fields
    header.set_field("Key0", "Value0");
    header.set_field("Key1", "Value1");
    header.set_field("Key2", "Value2");

    // AND WHEN removing the first added field
    header.remove_field("Key");

    // THEN the map contains only the latest added fields and not the first field
    ASSERT_FALSE(header.empty());
    ASSERT_TRUE(header.get_field("Key").empty());
    ASSERT_EQ("Value0", header.get_field("Key0"));
    ASSERT_EQ("Value1", header.get_field("Key1"));
    ASSERT_EQ("Value2", header.get_field("Key2"));
}

TEST(GenericHeader_GTest, remove_non_existing_field) {
    // GIVEN an empty header map
    Metavision::GenericHeader header;

    // THEN the header is empty
    ASSERT_TRUE(header.empty());

    // WHEN removing a non existing field
    // THEN no crash occur
    header.remove_field("NonExisting");
}

TEST(GenericHeader_GTest, get_non_existing_field) {
    // GIVEN an empty header map
    Metavision::GenericHeader header;

    // THEN the header is empty
    ASSERT_TRUE(header.empty());

    // WHEN retrieving a field that does not exist
    // THEN the retrieved field has no value
    ASSERT_EQ("", header.get_field("Key"));

    // AND THEN the header is still empty
    ASSERT_TRUE(header.empty());
}

TEST(GenericHeader_GTest, header_to_string) {
    // GIVEN an empty header map
    Metavision::GenericHeader header;

    // WHEN adding a field
    header.set_field("Key", "Value");

    // THEN the field and the value are found together when transforming to string
    std::string expected_field_line = "Key Value";
    ASSERT_NE(std::string::npos, header.to_string().find(expected_field_line));

    // WHEN adding other fields
    header.set_field("Key0", "Value0");
    header.set_field("Key1", "Value1");
    header.set_field("Key2", "Value2");

    // AND WHEN removing the first added field
    header.remove_field("Key");

    // THEN the header as string contains only the latest added fields
    ASSERT_EQ(std::string::npos, header.to_string().find(expected_field_line));
    expected_field_line = "Key0 Value0";
    ASSERT_NE(std::string::npos, header.to_string().find(expected_field_line));
    expected_field_line = "Key1 Value1";
    ASSERT_NE(std::string::npos, header.to_string().find(expected_field_line));
    expected_field_line = "Key2 Value2";
    ASSERT_NE(std::string::npos, header.to_string().find(expected_field_line));
}

TEST(GenericHeader_GTest, stream_with_empty_header) {
    // GIVEN a valid istream without a header
    std::stringstream ss;

    // WHEN building a generic header from it
    Metavision::GenericHeader header(ss);

    // THEN header map is empty
    ASSERT_TRUE(header.empty());
}

TEST(GenericHeader_GTest, stream_with_non_empty_header_and_set_new_field) {
    // GIVEN a valid istream with a header
    std::stringstream ss;
    ss << "% Key Value";

    // WHEN building a generic header from it
    Metavision::GenericHeader header(ss);

    // THEN header map is not empty
    ASSERT_FALSE(header.empty());

    // AND THEN the value in the header can be retrieved
    ASSERT_EQ("Value", header.get_field("Key"));

    // WHEN adding a new field
    header.set_field("OtherKey", "OtherValue");

    // THEN the value in the header can be retrieved
    ASSERT_EQ("OtherValue", header.get_field("OtherKey"));

    // AND WHEN converting the header to string
    // THEN all fields are found
    const std::string expected_field_line = "Key Value";
    ASSERT_NE(std::string::npos, header.to_string().find(expected_field_line));
    const std::string expected_other_field_line = "OtherKey OtherValue";
    ASSERT_NE(std::string::npos, header.to_string().find(expected_other_field_line));

    // WHEN removing the initial field
    header.remove_field("Key");

    // THEN the header doesn't contain it but contains the other field
    ASSERT_EQ(std::string::npos, header.to_string().find(expected_field_line));
    ASSERT_NE(std::string::npos, header.to_string().find(expected_other_field_line));
}

TEST(GenericHeader_GTest, input_header_empty) {
    // GIVEN an empty header type struct
    Metavision::GenericHeader::HeaderMap header_map;

    // WHEN building a generic header from it
    Metavision::GenericHeader header(header_map);

    // THEN header map is empty
    ASSERT_TRUE(header.empty());
}

TEST(GenericHeader_GTest, input_header_non_empty_and_set_new_field) {
    // GIVEN an empty header type struct
    Metavision::GenericHeader::HeaderMap header_map;
    header_map["Key"] = "Value";

    // WHEN building a generic header from it
    Metavision::GenericHeader header(header_map);

    // THEN header map is not empty
    ASSERT_FALSE(header.empty());

    // AND THEN the value in the header can be retrieved
    ASSERT_EQ("Value", header.get_field("Key"));

    // WHEN adding a new field
    header.set_field("OtherKey", "OtherValue");

    // THEN the value in the header can be retrieved
    ASSERT_EQ("OtherValue", header.get_field("OtherKey"));

    // AND WHEN converting the header to string
    // THEN all fields are found
    const std::string expected_field_line = "Key Value";
    ASSERT_NE(std::string::npos, header.to_string().find(expected_field_line));
    const std::string expected_other_field_line = "OtherKey OtherValue";
    ASSERT_NE(std::string::npos, header.to_string().find(expected_other_field_line));

    // WHEN removing the initial field
    header.remove_field("Key");

    // THEN the header doesn't contain it but contains the other field
    ASSERT_EQ(std::string::npos, header.to_string().find(expected_field_line));
    ASSERT_NE(std::string::npos, header.to_string().find(expected_other_field_line));
}

TEST(GenericHeader_GTest, stream_with_date) {
    // GIVEN a valid istream with a header containing a date
    std::stringstream ss;
    ss << "% Date Value";

    // WHEN building a generic header from it
    Metavision::GenericHeader header(ss);

    // THEN header map is not empty
    ASSERT_FALSE(header.empty());

    // AND THEN the date in the header can be retrieved
    ASSERT_EQ("Value", header.get_date());
}

TEST(GenericHeader_GTest, add_date) {
    // GIVEN an empty generic header
    Metavision::GenericHeader header;

    // THEN header map is empty
    ASSERT_TRUE(header.empty());

    // WHEN adding a date
    header.add_date();

    // THEN the date in the header can be retrieved and interpreted
    ASSERT_FALSE(header.get_date().empty());

    struct std::tm date;
    date.tm_year = -1;
    date.tm_mon  = -1;
    date.tm_mday = -1;
    date.tm_hour = -1;
    date.tm_min  = -1;
    date.tm_sec  = -1;

    std::istringstream read_date(header.get_date());
    read_date >> std::get_time(&date, "%Y-%m-%d %H:%M:%S");
    ASSERT_NE(-1, date.tm_year);
    ASSERT_NE(-1, date.tm_mon);
    ASSERT_NE(-1, date.tm_mday);
    ASSERT_NE(-1, date.tm_hour);
    ASSERT_NE(-1, date.tm_min);
    ASSERT_NE(-1, date.tm_sec);
}

TEST(GenericHeader_GTest, stream_with_end) {
    // GIVEN a valid istream with a header containing an explicit end marker
    std::stringstream ss;
    ss << "% Key1 Value1" << std::endl;
    ss << "% Key2 Value2" << std::endl;
    ss << "% Key3 Value3" << std::endl;
    ss << "% Key3 Value3" << std::endl;
    ss << "% end" << std::endl;
    ss << "% Key4 Value4" << std::endl;

    // WHEN building a generic header from it
    Metavision::GenericHeader header(ss);

    // THEN header map is not empty
    ASSERT_FALSE(header.empty());

    // AND THEN then values for Key1-3 can be retrieved but not Key4
    ASSERT_EQ("Value1", header.get_field("Key1"));
    ASSERT_EQ("Value2", header.get_field("Key2"));
    ASSERT_EQ("Value3", header.get_field("Key3"));
    ASSERT_EQ("", header.get_field("end"));
    ASSERT_EQ("", header.get_field("Key4"));
}

class GenericHeaderWithFile_GTest : public Metavision::GTestWithTmpDir {
protected:
    virtual void SetUp() override {
        path_ = tmpdir_handler_->get_full_path("rawfile.raw");
    }
    std::string path_;
};

TEST_F(GenericHeaderWithFile_GTest, parse_strictly_the_header_and_nothing_more) {
    std::ofstream header_file(path_, std::ios::binary);
    std::string str_to_write = "% Key Value\n";
    header_file << str_to_write;
    std::vector<int> written(4, 123);
    header_file.write(reinterpret_cast<char *>(written.data()), written.size() * sizeof(int));
    header_file.close();

    std::ifstream read_file(path_, std::ios::binary);
    ASSERT_TRUE(read_file.is_open());
    Metavision::GenericHeader header(read_file);
    std::vector<int> read(4);
    read_file.read(reinterpret_cast<char *>(read.data()), written.size() * sizeof(int));
    ASSERT_EQ(written, read);
}
