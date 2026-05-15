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

#include <gtest/gtest.h>
#include <atomic>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/utils/frame_composer.h"

using namespace Metavision;

class FrameComposer_GTest : public ::testing::Test {
public:
    FrameComposer_GTest() {}

    virtual ~FrameComposer_GTest() {}

protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}
};

TEST_F(FrameComposer_GTest, fit_size) {
    // GIVEN 6 images arranged like below
    //
    // +-------------+-------------+--------------------+
    // |             |             |                    |
    // |    Img 1    |    Img 2    |                    |
    // |             |             |                    |
    // +-------------+-------------+       Img 5        |
    // |                           |                    |
    // |          Img 3            |                    |
    // |                           |                    |
    // +---------------------------+-------+------------+
    // |                                   |            |
    // |                                   |            |
    // |                                   |  +------+  |
    // |               Img 4               |  | Img 6|  |
    // |                                   |  +------+  |
    // |                                   |            |
    // +-----------------------------------+------------+
    //

    std::vector<cv::Mat> frames;
    std::vector<cv::Rect> rois;
    frames.reserve(6);
    rois.reserve(6);

    int id             = 127;
    auto add_ref_image = [&](int left_x, int top_y, int width, int height) {
        frames.emplace_back(width, height, CV_8UC3, cv::Scalar::all(id++));
        rois.emplace_back(left_x, top_y, width, height);
    };

    add_ref_image(0, 0, 10, 10);  // Img 1
    add_ref_image(10, 0, 10, 10); // Img 2
    add_ref_image(0, 10, 20, 10); // Img 3
    add_ref_image(0, 20, 25, 20); // Img 4
    add_ref_image(20, 0, 20, 20); // Img 5
    add_ref_image(27, 22, 2, 3);  // Img 6

    // WHEN we add them to the FrameComposer
    FrameComposer composer(cv::Vec3b(0, 0, 0));
    std::vector<std::pair<int, int>> sizes;
    sizes.reserve(frames.size());
    using SizeType = std::vector<cv::Mat>::size_type;
    for (SizeType i = 0; i < frames.size(); ++i) {
        const cv::Rect &roi = rois[i];
        FrameComposer::ResizingOptions resize_options(roi.width, roi.height);
        FrameComposer::GrayToColorOptions gray_o;
        const unsigned int id = composer.add_new_subimage_parameters(roi.x, roi.y, resize_options, gray_o);
        composer.update_subimage(id, frames[i]);
        sizes.emplace_back(composer.get_total_width(), composer.get_total_height());
    }

    // THEN we get the expected sizes for the composed image after each update
    ASSERT_EQ(sizes.size(), 6);
    ASSERT_EQ(sizes[0], std::make_pair(10, 10));
    ASSERT_EQ(sizes[1], std::make_pair(20, 10));
    ASSERT_EQ(sizes[2], std::make_pair(20, 20));
    ASSERT_EQ(sizes[3], std::make_pair(25, 40));
    ASSERT_EQ(sizes[4], std::make_pair(40, 40));
    ASSERT_EQ(sizes[5], std::make_pair(40, 40));
}

TEST_F(FrameComposer_GTest, compose_color_images) {
    // GIVEN three monochromatic images of the same size: Blue, Green and Red
    cv::Mat ref_frame(20, 90, CV_8UC3, cv::Vec3b(0, 0, 0));
    const int width = 30, height = 20;
    std::vector<cv::Mat> frames;
    for (int i = 0; i < 3; ++i) {
        auto color = cv::Vec3b(i == 0 ? 255 : 0, i == 1 ? 255 : 0, i == 2 ? 255 : 0);
        frames.emplace_back(height, width, CV_8UC3, color);
        cv::Rect roi(width * i, 0, width, height);
        ref_frame(roi).setTo(color);
    }

    // WHEN we add them to the FrameComposer
    FrameComposer composer(cv::Vec3b(0, 0, 0));

    for (int i = 0; i < 3; ++i) {
        FrameComposer::ResizingOptions resize_options(width, height);
        FrameComposer::GrayToColorOptions gray_o;
        const unsigned int id = composer.add_new_subimage_parameters(width * i, 0, resize_options, gray_o);
        composer.update_subimage(id, frames[i]);
    }

    // THEN we get the expected composed image, which is the horizontal concatenation of the input images
    const cv::Mat &test_frame = composer.get_full_image();
    ASSERT_EQ(ref_frame.size(), test_frame.size());
    ASSERT_EQ(ref_frame.type(), test_frame.type());
    bool is_equal = true;
    for (int y = 0; y < test_frame.rows; ++y) {
        for (int x = 0; x < test_frame.cols; ++x) {
            if (ref_frame.at<cv::Vec3b>(y, x) != test_frame.at<cv::Vec3b>(y, x)) {
                MV_SDK_LOG_ERROR() << x << " " << y << " " << ref_frame.at<cv::Vec3b>(y, x) << " "
                                   << test_frame.at<cv::Vec3b>(y, x) << test_frame.size();
                is_equal = false;
            }
        }
    }
    ASSERT_TRUE(is_equal);
}

TEST_F(FrameComposer_GTest, compose_resize_color_images) {
    // GIVEN three monochromatic images of the same size: Blue, Green and Red
    cv::Mat ref_frame(5, 45, CV_8UC3, cv::Vec3b(0, 0, 0));
    const int width = 3, height = 20;
    const int resized_width = 15, resized_height = 5;
    std::vector<cv::Mat> frames;
    for (int i = 0; i < 3; ++i) {
        auto color = cv::Vec3b(i == 0 ? 255 : 0, i == 1 ? 255 : 0, i == 2 ? 255 : 0);
        frames.emplace_back(height, width, CV_8UC3, color);
        cv::Rect roi(resized_width * i, 0, resized_width, resized_height);
        ref_frame(roi).setTo(color);
    }

    // WHEN we add them to the FrameComposer, with the resize option
    FrameComposer composer(cv::Vec3b(0, 0, 0));

    for (int i = 0; i < 3; ++i) {
        FrameComposer::ResizingOptions resize_options(resized_width, resized_height);
        FrameComposer::GrayToColorOptions gray_o;
        const unsigned int id = composer.add_new_subimage_parameters(resized_width * i, 0, resize_options, gray_o);
        composer.update_subimage(id, frames[i]);
    }

    // THEN we get the expected composed image, which is the horizontal concatenation of the resized input images
    const cv::Mat &test_frame = composer.get_full_image();
    ASSERT_EQ(ref_frame.size(), test_frame.size());
    ASSERT_EQ(ref_frame.type(), test_frame.type());
    bool is_equal = true;
    for (int y = 0; y < test_frame.rows; ++y) {
        for (int x = 0; x < test_frame.cols; ++x) {
            if (ref_frame.at<cv::Vec3b>(y, x) != test_frame.at<cv::Vec3b>(y, x)) {
                MV_SDK_LOG_ERROR() << x << " " << y << " " << ref_frame.at<cv::Vec3b>(y, x) << " "
                                   << test_frame.at<cv::Vec3b>(y, x) << test_frame.size();
                is_equal = false;
            }
        }
    }
    ASSERT_TRUE(is_equal);
}

TEST_F(FrameComposer_GTest, compose_resize_color_float_images) {
    // GIVEN three monochromatic images of the same size: Blue, Green and Red
    // using floating point pixels inside [0,1]
    cv::Mat ref_frame(5, 45, CV_8UC3, cv::Vec3b(0, 0, 0));
    const int width = 3, height = 20;
    const int resized_width = 15, resized_height = 5;
    std::vector<cv::Mat> frames;
    for (int i = 0; i < 3; ++i) {
        auto color = cv::Scalar(i == 0 ? 1. : 0., i == 1 ? 1. : 0., i == 2 ? 1. : 0.);
        frames.emplace_back(height, width, CV_64FC3, color); // Double
        cv::Rect roi(resized_width * i, 0, resized_width, resized_height);
        ref_frame(roi).setTo(255 * color);
    }

    // WHEN we add them to the FrameComposer, with the resize option
    FrameComposer composer(cv::Vec3b(0, 0, 0));

    for (int i = 0; i < 3; ++i) {
        FrameComposer::ResizingOptions resize_options(resized_width, resized_height);
        FrameComposer::GrayToColorOptions gray_o;
        const unsigned int id = composer.add_new_subimage_parameters(resized_width * i, 0, resize_options, gray_o);
        composer.update_subimage(id, frames[i]);
    }

    // THEN we get the expected composed image, which is the horizontal concatenation of the resized input images
    const cv::Mat &test_frame = composer.get_full_image();
    ASSERT_EQ(ref_frame.size(), test_frame.size());
    ASSERT_EQ(ref_frame.type(), test_frame.type());
    bool is_equal = true;
    for (int y = 0; y < test_frame.rows; ++y) {
        for (int x = 0; x < test_frame.cols; ++x) {
            if (ref_frame.at<cv::Vec3b>(y, x) != test_frame.at<cv::Vec3b>(y, x)) {
                MV_SDK_LOG_ERROR() << x << " " << y << " " << ref_frame.at<cv::Vec3b>(y, x) << " "
                                   << test_frame.at<cv::Vec3b>(y, x) << test_frame.size();
                is_equal = false;
            }
        }
    }
    ASSERT_TRUE(is_equal);
}

TEST_F(FrameComposer_GTest, resize_and_crop_color_images) {
    // GIVEN an image with a white background and a colored filled rectangle drawn on the center
    const int width = 30, height = 20;
    const int colored_width = 10, colored_height = 4;
    cv::Mat input_frame(height, width, CV_8UC3, cv::Vec3b(255, 255, 255));
    cv::Rect color_roi((width - colored_width) / 2, (height - colored_height) / 2, colored_width, colored_height);
    input_frame(color_roi).setTo(cv::Vec3b(125, 15, 250));

    // WHEN we add it twice to the FrameComposer, but first with the cropping mode
    // and then with the standard resizing mode
    FrameComposer composer(cv::Vec3b(0, 0, 0));
    const int resized_width = 15, resized_height = 5;
    for (int i = 0; i < 2; ++i) {
        FrameComposer::ResizingOptions resize_options(resized_width, resized_height, i == 0,
                                                      FrameComposer::InterpolationType::Nearest);
        FrameComposer::GrayToColorOptions gray_o;
        const unsigned int id = composer.add_new_subimage_parameters(resized_width * i, 0, resize_options, gray_o);
        composer.update_subimage(id, input_frame);
    }

    // THEN we get the expected composed image, which is the horizontal concatenation of the cropped and resized
    // versions of the input image
    cv::Rect crop_roi((width - resized_width) / 2, (height - resized_height) / 2, resized_width, resized_height);
    cv::Mat ref_right;
    cv::resize(input_frame, ref_right, cv::Size(resized_width, resized_height), 0., 0., cv::INTER_NEAREST);
    cv::Mat ref_frame;
    cv::hconcat(input_frame(crop_roi), ref_right, ref_frame);

    const cv::Mat &test_frame = composer.get_full_image();
    ASSERT_EQ(ref_frame.size(), test_frame.size());
    ASSERT_EQ(ref_frame.type(), test_frame.type());
    bool is_equal = true;
    for (int y = 0; y < test_frame.rows; ++y) {
        for (int x = 0; x < test_frame.cols; ++x) {
            if (ref_frame.at<cv::Vec3b>(y, x) != test_frame.at<cv::Vec3b>(y, x)) {
                MV_SDK_LOG_ERROR() << x << " " << y << " " << ref_frame.at<cv::Vec3b>(y, x) << " "
                                   << test_frame.at<cv::Vec3b>(y, x) << test_frame.size();
                is_equal = false;
            }
        }
    }
    ASSERT_TRUE(is_equal);
}

TEST_F(FrameComposer_GTest, compose_grey_images) {
    // GIVEN three monochromatic gray images of the same size
    cv::Mat ref_frame(20, 90, CV_8UC3, cv::Vec3b(0, 0, 0));
    const int width = 30, height = 20;
    std::vector<cv::Mat> frames;
    for (int i = 0; i < 3; ++i) {
        char intensity(40 * i);
        frames.emplace_back(height, width, CV_8UC1, cv::Scalar(intensity));
        cv::Rect roi(width * i, 0, width, height);
        ref_frame(roi).setTo(cv::Scalar::all(intensity));
    }

    // WHEN we add them to the FrameComposer
    FrameComposer composer(cv::Vec3b(0, 0, 0));
    for (int i = 0; i < 3; ++i) {
        FrameComposer::ResizingOptions resize_options(width, height);
        FrameComposer::GrayToColorOptions gray_o;
        const unsigned int id = composer.add_new_subimage_parameters(width * i, 0, resize_options, gray_o);
        composer.update_subimage(id, frames[i]);
    }

    // THEN we get the expected composed image, which is the horizontal concatenation of the input images
    const cv::Mat &test_frame = composer.get_full_image();
    ASSERT_EQ(ref_frame.size(), test_frame.size());
    ASSERT_EQ(ref_frame.type(), test_frame.type());
    bool is_equal = true;
    for (int y = 0; y < test_frame.rows; ++y) {
        for (int x = 0; x < test_frame.cols; ++x) {
            if (ref_frame.at<cv::Vec3b>(y, x) != test_frame.at<cv::Vec3b>(y, x)) {
                MV_SDK_LOG_ERROR() << x << " " << y << " " << ref_frame.at<cv::Vec3b>(y, x) << " "
                                   << test_frame.at<cv::Vec3b>(y, x) << test_frame.size();
                is_equal = false;
            }
        }
    }
    ASSERT_TRUE(is_equal);
}

TEST_F(FrameComposer_GTest, compose_grey_float_images) {
    // GIVEN three monochromatic gray images of the same size
    cv::Mat ref_frame(20, 90, CV_8UC3, cv::Vec3b(0, 0, 0));
    const int width = 30, height = 20;
    std::vector<cv::Mat> frames;
    for (int i = 0; i < 3; ++i) {
        char intensity(40 * i);
        frames.emplace_back(height, width, CV_64FC1, cv::Scalar(intensity / 255.0)); // Double
        cv::Rect roi(width * i, 0, width, height);
        ref_frame(roi).setTo(cv::Scalar::all(intensity));
    }

    // WHEN we add them to the FrameComposer
    FrameComposer composer(cv::Vec3b(0, 0, 0));
    for (int i = 0; i < 3; ++i) {
        FrameComposer::ResizingOptions resize_options(width, height);
        FrameComposer::GrayToColorOptions gray_o;
        const unsigned int id = composer.add_new_subimage_parameters(width * i, 0, resize_options, gray_o);
        composer.update_subimage(id, frames[i]);
    }

    // THEN we get the expected composed image, which is the horizontal concatenation of the input images
    const cv::Mat &test_frame = composer.get_full_image();
    ASSERT_EQ(ref_frame.size(), test_frame.size());
    ASSERT_EQ(ref_frame.type(), test_frame.type());
    bool is_equal = true;
    for (int y = 0; y < test_frame.rows; ++y) {
        for (int x = 0; x < test_frame.cols; ++x) {
            if (ref_frame.at<cv::Vec3b>(y, x) != test_frame.at<cv::Vec3b>(y, x)) {
                MV_SDK_LOG_ERROR() << x << " " << y << " " << ref_frame.at<cv::Vec3b>(y, x) << " "
                                   << test_frame.at<cv::Vec3b>(y, x) << test_frame.size();
                is_equal = false;
            }
        }
    }
    ASSERT_TRUE(is_equal);
}

TEST_F(FrameComposer_GTest, gray_to_color) {
    // GIVEN a gray image composed of 3 vertical stripes of the same width
    const int width = 12, height = 5;
    const int width_row = 4;
    cv::Mat input_striped_frame(height, width, CV_8UC1, cv::Scalar(0));
    cv::Mat rgb_striped_frame(height, width, CV_8UC3, cv::Vec3b(0, 0, 0));
    for (int i = 0; i < 3; ++i) {
        cv::Rect roi(width_row * i, 0, width_row, height);
        const uchar intensity = 20 * i + 45;
        input_striped_frame(roi).setTo(cv::Scalar(intensity));
        rgb_striped_frame(roi).setTo(cv::Scalar::all(intensity));
    }

    // WHEN we add it to the FrameComposer using different rescaling and colormap options
    FrameComposer composer(cv::Vec3b(0, 0, 0));
    int k = 0;
    FrameComposer::ResizingOptions resize_options(width, height);
    auto add_image_to_the_composer = [&](unsigned char min_val, unsigned char max_val, int map_id) {
        FrameComposer::GrayToColorOptions gray_o;

        gray_o.min_rescaling_value = min_val;
        gray_o.max_rescaling_value = max_val;
        gray_o.color_map_id        = map_id;
        unsigned int id            = composer.add_new_subimage_parameters(width * (k++), 0, resize_options, gray_o);
        composer.update_subimage(id, input_striped_frame);
    };

    add_image_to_the_composer(0, 255, -1); // 1) Same image
    add_image_to_the_composer(45, 45, -1); // 2) Same image
    add_image_to_the_composer(45, 85, -1); // 3) Rescaled
    add_image_to_the_composer(0, 255, 0);  // 4) Colored
    add_image_to_the_composer(45, 85, 0);  // 5) Rescaled and colored

    // THEN we get the expected composed image, which is the horizontal concatenation of the modified versions of the
    // input image
    std::vector<cv::Mat> frames        = {rgb_striped_frame, rgb_striped_frame};
    auto add_ref_colored_striped_image = [&](const std::array<cv::Vec3b, 3> &color_rois) {
        frames.emplace_back();
        cv::Mat &mat = frames.back();
        mat.create(height, width, CV_8UC3);
        mat.setTo(cv::Vec3b(0, 0, 0));
        for (int i = 0; i < 3; ++i) {
            const cv::Rect roi(width_row * i, 0, width_row, height);
            mat(roi).setTo(color_rois[i]);
        }
    };
    add_ref_colored_striped_image(std::array<cv::Vec3b, 3>{{{0, 0, 0}, {127, 127, 127}, {255, 255, 255}}});
    add_ref_colored_striped_image(std::array<cv::Vec3b, 3>{{{0, 45, 255}, {0, 65, 255}, {0, 85, 255}}});
    add_ref_colored_striped_image(std::array<cv::Vec3b, 3>{{{0, 0, 255}, {0, 127, 255}, {0, 255, 255}}});

    cv::Mat ref_frame;
    cv::hconcat(frames, ref_frame);

    const cv::Mat &test_frame = composer.get_full_image();
    ASSERT_EQ(ref_frame.size(), test_frame.size());
    ASSERT_EQ(ref_frame.type(), test_frame.type());
    bool is_equal = true;
    for (int y = 0; y < test_frame.rows; ++y) {
        for (int x = 0; x < test_frame.cols; ++x) {
            if (ref_frame.at<cv::Vec3b>(y, x) != test_frame.at<cv::Vec3b>(y, x)) {
                MV_SDK_LOG_ERROR() << x << " " << y << " " << ref_frame.at<cv::Vec3b>(y, x) << " "
                                   << test_frame.at<cv::Vec3b>(y, x) << test_frame.size();
                is_equal = false;
            }
        }
    }
    ASSERT_TRUE(is_equal);
}
