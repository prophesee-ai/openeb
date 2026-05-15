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
#include <opencv2/imgproc.hpp>

#include "metavision/sdk/core/utils/cv_color_map.h"

using namespace Metavision;

class CvColorMap_GTest : public ::testing::Test {
public:
    CvColorMap_GTest() {}

    virtual ~CvColorMap_GTest() {}

protected:
    virtual void SetUp() override {}

    virtual void TearDown() override {}
};

TEST(CvColorMap_GTest, gray) {
    // GIVEN three grayscale images and an instance of CvColorMap
    const int width  = 10;
    const int height = 13;
    std::vector<cv::Mat> gray_images;
    gray_images.emplace_back(height, width, CV_8UC1, 0);
    gray_images.emplace_back(height, width, CV_8UC1, 255);
    gray_images.emplace_back(height, width, CV_8UC1);
    for (int i = 0; i < height; i++)
        for (int j = 0; j < height; j++)
            gray_images.back().at<uchar>(i, j) = i * 11 + j;

    CvColorMap cm(cv::COLORMAP_JET);

    // WHEN we apply both CvColorMap and cv::applyColorMap
    std::vector<cv::Mat> color_test;
    std::vector<cv::Mat> color_gt;
    for (const cv::Mat &img : gray_images) {
        color_test.emplace_back();
        cm(img, color_test.back());

        color_gt.emplace_back();
        cv::applyColorMap(img, color_gt.back(), cv::COLORMAP_JET);
    }

    // THEN we generate the same color images
    ASSERT_EQ(3, color_test.size());
    ASSERT_EQ(3, color_gt.size());
    for (int k = 0; k < 3; k++) {
        ASSERT_EQ(color_gt[k].size(), color_gt[k].size());
        ASSERT_EQ(CV_8UC3, color_gt[k].type());
        ASSERT_TRUE(
            std::equal(color_gt[k].begin<cv::Vec3b>(), color_gt[k].end<cv::Vec3b>(), color_test[k].begin<cv::Vec3b>()));
    }
}

TEST(CvColorMap_GTest, color) {
    // GIVEN three color images and an instance of CvColorMap
    const int width  = 10;
    const int height = 13;
    std::vector<cv::Mat> color_images;
    color_images.emplace_back(height, width, CV_8UC3, cv::Vec3b::all(0));
    color_images.emplace_back(height, width, CV_8UC3, cv::Vec3b(17, 185, 96));
    color_images.emplace_back(height, width, CV_8UC3);
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            color_images.back().at<cv::Vec3b>(i, j) = cv::Vec3b(i + 4 * j, i + 15 * j, 8 * i + j);

    CvColorMap cm(cv::COLORMAP_JET);

    // WHEN we apply both CvColorMap and cv::applyColorMap
    std::vector<cv::Mat> color_test;
    std::vector<cv::Mat> color_gt;
    for (const cv::Mat &img : color_images) {
        color_test.emplace_back();
        cm(img, color_test.back());

        color_gt.emplace_back();
        cv::applyColorMap(img, color_gt.back(), cv::COLORMAP_JET);
    }

    // THEN we generate the same color images
    ASSERT_EQ(3, color_test.size());
    ASSERT_EQ(3, color_gt.size());
    for (int k = 0; k < 3; k++) {
        ASSERT_EQ(color_gt[k].size(), color_gt[k].size());
        ASSERT_EQ(CV_8UC3, color_gt[k].type());
        ASSERT_TRUE(
            std::equal(color_gt[k].begin<cv::Vec3b>(), color_gt[k].end<cv::Vec3b>(), color_test[k].begin<cv::Vec3b>()));
    }
}
