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

#ifndef METAVISION_SDK_ANALYTICS_RANDOM_COLOR_MAP_H
#define METAVISION_SDK_ANALYTICS_RANDOM_COLOR_MAP_H

#include <opencv2/opencv.hpp>

/// Total number of colors
static const int N_COLORS = 256;

#define RGBu_TO_RGBf(r, g, b) cv::Scalar(r, g, b)
/// Array of N_COLORS different colors
static cv::Scalar COLORS[N_COLORS] = {
    RGBu_TO_RGBf(0, 0, 255),     RGBu_TO_RGBf(255, 0, 0),     RGBu_TO_RGBf(0, 255, 0),     RGBu_TO_RGBf(255, 0, 182),
    RGBu_TO_RGBf(0, 83, 0),      RGBu_TO_RGBf(255, 211, 0),   RGBu_TO_RGBf(0, 159, 255),   RGBu_TO_RGBf(154, 77, 66),
    RGBu_TO_RGBf(0, 255, 190),   RGBu_TO_RGBf(120, 63, 193),  RGBu_TO_RGBf(31, 150, 152),  RGBu_TO_RGBf(255, 172, 253),
    RGBu_TO_RGBf(177, 204, 113), RGBu_TO_RGBf(241, 8, 92),    RGBu_TO_RGBf(254, 143, 66),  RGBu_TO_RGBf(221, 0, 255),
    RGBu_TO_RGBf(114, 0, 85),    RGBu_TO_RGBf(118, 108, 149), RGBu_TO_RGBf(2, 173, 36),    RGBu_TO_RGBf(200, 255, 0),
    RGBu_TO_RGBf(136, 108, 0),   RGBu_TO_RGBf(255, 183, 159), RGBu_TO_RGBf(133, 133, 103), RGBu_TO_RGBf(161, 3, 0),
    RGBu_TO_RGBf(20, 249, 255),  RGBu_TO_RGBf(0, 71, 158),    RGBu_TO_RGBf(220, 94, 147),  RGBu_TO_RGBf(147, 212, 255),
    RGBu_TO_RGBf(0, 76, 255),    RGBu_TO_RGBf(0, 66, 80),     RGBu_TO_RGBf(57, 167, 106),  RGBu_TO_RGBf(238, 112, 254),
    RGBu_TO_RGBf(0, 0, 100),     RGBu_TO_RGBf(171, 245, 204), RGBu_TO_RGBf(161, 146, 255), RGBu_TO_RGBf(164, 255, 115),
    RGBu_TO_RGBf(255, 206, 113), RGBu_TO_RGBf(71, 0, 21),     RGBu_TO_RGBf(212, 173, 197), RGBu_TO_RGBf(251, 118, 111),
    RGBu_TO_RGBf(171, 188, 0),   RGBu_TO_RGBf(117, 0, 215),   RGBu_TO_RGBf(166, 0, 154),   RGBu_TO_RGBf(0, 115, 254),
    RGBu_TO_RGBf(165, 93, 174),  RGBu_TO_RGBf(98, 132, 2),    RGBu_TO_RGBf(0, 121, 168),   RGBu_TO_RGBf(0, 255, 131),
    RGBu_TO_RGBf(86, 53, 0),     RGBu_TO_RGBf(159, 0, 63),    RGBu_TO_RGBf(66, 45, 66),    RGBu_TO_RGBf(255, 242, 187),
    RGBu_TO_RGBf(0, 93, 67),     RGBu_TO_RGBf(252, 255, 124), RGBu_TO_RGBf(159, 191, 186), RGBu_TO_RGBf(167, 84, 19),
    RGBu_TO_RGBf(74, 39, 108),   RGBu_TO_RGBf(0, 16, 166),    RGBu_TO_RGBf(145, 78, 109),  RGBu_TO_RGBf(207, 149, 0),
    RGBu_TO_RGBf(195, 187, 255), RGBu_TO_RGBf(253, 68, 64),   RGBu_TO_RGBf(66, 78, 32),    RGBu_TO_RGBf(106, 1, 0),
    RGBu_TO_RGBf(181, 131, 84),  RGBu_TO_RGBf(132, 233, 147), RGBu_TO_RGBf(96, 217, 0),    RGBu_TO_RGBf(255, 111, 211),
    RGBu_TO_RGBf(102, 75, 63),   RGBu_TO_RGBf(254, 100, 0),   RGBu_TO_RGBf(228, 3, 127),   RGBu_TO_RGBf(17, 199, 174),
    RGBu_TO_RGBf(210, 129, 139), RGBu_TO_RGBf(91, 118, 124),  RGBu_TO_RGBf(32, 59, 106),   RGBu_TO_RGBf(180, 84, 255),
    RGBu_TO_RGBf(226, 8, 210),   RGBu_TO_RGBf(0, 1, 20),      RGBu_TO_RGBf(93, 132, 68),   RGBu_TO_RGBf(166, 250, 255),
    RGBu_TO_RGBf(97, 123, 201),  RGBu_TO_RGBf(98, 0, 122),    RGBu_TO_RGBf(126, 190, 58),  RGBu_TO_RGBf(0, 60, 183),
    RGBu_TO_RGBf(255, 253, 0),   RGBu_TO_RGBf(7, 197, 226),   RGBu_TO_RGBf(180, 167, 57),  RGBu_TO_RGBf(148, 186, 138),
    RGBu_TO_RGBf(204, 187, 160), RGBu_TO_RGBf(55, 0, 49),     RGBu_TO_RGBf(0, 40, 1),      RGBu_TO_RGBf(150, 122, 129),
    RGBu_TO_RGBf(39, 136, 38),   RGBu_TO_RGBf(206, 130, 180), RGBu_TO_RGBf(150, 164, 196), RGBu_TO_RGBf(180, 32, 128),
    RGBu_TO_RGBf(110, 86, 180),  RGBu_TO_RGBf(147, 0, 185),   RGBu_TO_RGBf(199, 48, 61),   RGBu_TO_RGBf(115, 102, 255),
    RGBu_TO_RGBf(15, 187, 253),  RGBu_TO_RGBf(172, 164, 100), RGBu_TO_RGBf(182, 117, 250), RGBu_TO_RGBf(216, 220, 254),
    RGBu_TO_RGBf(87, 141, 113),  RGBu_TO_RGBf(216, 85, 34),   RGBu_TO_RGBf(0, 196, 103),   RGBu_TO_RGBf(243, 165, 105),
    RGBu_TO_RGBf(216, 255, 182), RGBu_TO_RGBf(1, 24, 219),    RGBu_TO_RGBf(52, 66, 54),    RGBu_TO_RGBf(255, 154, 0),
    RGBu_TO_RGBf(87, 95, 1),     RGBu_TO_RGBf(198, 241, 79),  RGBu_TO_RGBf(255, 95, 133),  RGBu_TO_RGBf(123, 172, 240),
    RGBu_TO_RGBf(120, 100, 49),  RGBu_TO_RGBf(162, 133, 204), RGBu_TO_RGBf(105, 255, 220), RGBu_TO_RGBf(198, 82, 100),
    RGBu_TO_RGBf(121, 26, 64),   RGBu_TO_RGBf(0, 238, 70),    RGBu_TO_RGBf(231, 207, 69),  RGBu_TO_RGBf(217, 128, 233),
    RGBu_TO_RGBf(255, 211, 209), RGBu_TO_RGBf(209, 255, 141), RGBu_TO_RGBf(36, 0, 3),      RGBu_TO_RGBf(87, 163, 193),
    RGBu_TO_RGBf(211, 231, 201), RGBu_TO_RGBf(203, 111, 79),  RGBu_TO_RGBf(62, 24, 0),     RGBu_TO_RGBf(0, 117, 223),
    RGBu_TO_RGBf(112, 176, 88),  RGBu_TO_RGBf(209, 24, 0),    RGBu_TO_RGBf(0, 30, 107),    RGBu_TO_RGBf(105, 200, 197),
    RGBu_TO_RGBf(255, 203, 255), RGBu_TO_RGBf(233, 194, 137), RGBu_TO_RGBf(191, 129, 46),  RGBu_TO_RGBf(69, 42, 145),
    RGBu_TO_RGBf(171, 76, 194),  RGBu_TO_RGBf(14, 117, 61),   RGBu_TO_RGBf(0, 30, 25),     RGBu_TO_RGBf(118, 73, 127),
    RGBu_TO_RGBf(255, 169, 200), RGBu_TO_RGBf(94, 55, 217),   RGBu_TO_RGBf(238, 230, 138), RGBu_TO_RGBf(159, 54, 33),
    RGBu_TO_RGBf(80, 0, 148),    RGBu_TO_RGBf(189, 144, 128), RGBu_TO_RGBf(0, 109, 126),   RGBu_TO_RGBf(88, 223, 96),
    RGBu_TO_RGBf(71, 80, 103),   RGBu_TO_RGBf(1, 93, 159),    RGBu_TO_RGBf(99, 48, 60),    RGBu_TO_RGBf(2, 206, 148),
    RGBu_TO_RGBf(139, 83, 37),   RGBu_TO_RGBf(171, 0, 255),   RGBu_TO_RGBf(141, 42, 135),  RGBu_TO_RGBf(85, 83, 148),
    RGBu_TO_RGBf(150, 255, 0),   RGBu_TO_RGBf(0, 152, 123),   RGBu_TO_RGBf(255, 138, 203), RGBu_TO_RGBf(222, 69, 200),
    RGBu_TO_RGBf(107, 109, 230), RGBu_TO_RGBf(30, 0, 68),     RGBu_TO_RGBf(173, 76, 138),  RGBu_TO_RGBf(255, 134, 161),
    RGBu_TO_RGBf(0, 35, 60),     RGBu_TO_RGBf(138, 205, 0),   RGBu_TO_RGBf(111, 202, 157), RGBu_TO_RGBf(225, 75, 253),
    RGBu_TO_RGBf(255, 176, 77),  RGBu_TO_RGBf(229, 232, 57),  RGBu_TO_RGBf(114, 16, 255),  RGBu_TO_RGBf(111, 82, 101),
    RGBu_TO_RGBf(134, 137, 48),  RGBu_TO_RGBf(99, 38, 80),    RGBu_TO_RGBf(105, 38, 32),   RGBu_TO_RGBf(200, 110, 0),
    RGBu_TO_RGBf(209, 164, 255), RGBu_TO_RGBf(198, 210, 86),  RGBu_TO_RGBf(79, 103, 77),   RGBu_TO_RGBf(174, 165, 166),
    RGBu_TO_RGBf(170, 45, 101),  RGBu_TO_RGBf(199, 81, 175),  RGBu_TO_RGBf(255, 89, 172),  RGBu_TO_RGBf(146, 102, 78),
    RGBu_TO_RGBf(102, 134, 184), RGBu_TO_RGBf(111, 152, 255), RGBu_TO_RGBf(92, 255, 159),  RGBu_TO_RGBf(172, 137, 178),
    RGBu_TO_RGBf(210, 34, 98),   RGBu_TO_RGBf(199, 207, 147), RGBu_TO_RGBf(255, 185, 30),  RGBu_TO_RGBf(250, 148, 141),
    RGBu_TO_RGBf(49, 34, 78),    RGBu_TO_RGBf(254, 81, 97),   RGBu_TO_RGBf(254, 141, 100), RGBu_TO_RGBf(68, 54, 23),
    RGBu_TO_RGBf(201, 162, 84),  RGBu_TO_RGBf(199, 232, 240), RGBu_TO_RGBf(68, 152, 0),    RGBu_TO_RGBf(147, 172, 58),
    RGBu_TO_RGBf(22, 75, 28),    RGBu_TO_RGBf(8, 84, 121),    RGBu_TO_RGBf(116, 45, 0),    RGBu_TO_RGBf(104, 60, 255),
    RGBu_TO_RGBf(64, 41, 38),    RGBu_TO_RGBf(164, 113, 215), RGBu_TO_RGBf(207, 0, 155),   RGBu_TO_RGBf(118, 1, 35),
    RGBu_TO_RGBf(83, 0, 88),     RGBu_TO_RGBf(0, 82, 232),    RGBu_TO_RGBf(43, 92, 87),    RGBu_TO_RGBf(160, 217, 146),
    RGBu_TO_RGBf(176, 26, 229),  RGBu_TO_RGBf(29, 3, 36),     RGBu_TO_RGBf(122, 58, 159),  RGBu_TO_RGBf(214, 209, 207),
    RGBu_TO_RGBf(160, 100, 105), RGBu_TO_RGBf(106, 157, 160), RGBu_TO_RGBf(153, 219, 113), RGBu_TO_RGBf(192, 56, 207),
    RGBu_TO_RGBf(125, 255, 89),  RGBu_TO_RGBf(149, 0, 34),    RGBu_TO_RGBf(213, 162, 223), RGBu_TO_RGBf(22, 131, 204),
    RGBu_TO_RGBf(166, 249, 69),  RGBu_TO_RGBf(109, 105, 97),  RGBu_TO_RGBf(86, 188, 78),   RGBu_TO_RGBf(255, 109, 81),
    RGBu_TO_RGBf(255, 3, 248),   RGBu_TO_RGBf(255, 0, 73),    RGBu_TO_RGBf(202, 0, 35),    RGBu_TO_RGBf(67, 109, 18),
    RGBu_TO_RGBf(234, 170, 173), RGBu_TO_RGBf(191, 165, 0),   RGBu_TO_RGBf(38, 44, 51),    RGBu_TO_RGBf(85, 185, 2),
    RGBu_TO_RGBf(121, 182, 158), RGBu_TO_RGBf(254, 236, 212), RGBu_TO_RGBf(139, 165, 89),  RGBu_TO_RGBf(141, 254, 193),
    RGBu_TO_RGBf(0, 60, 43),     RGBu_TO_RGBf(63, 17, 40),    RGBu_TO_RGBf(255, 221, 246), RGBu_TO_RGBf(17, 26, 146),
    RGBu_TO_RGBf(32, 26, 1),     RGBu_TO_RGBf(154, 66, 84),   RGBu_TO_RGBf(149, 157, 238), RGBu_TO_RGBf(126, 130, 72),
    RGBu_TO_RGBf(58, 6, 101),    RGBu_TO_RGBf(189, 117, 101), RGBu_TO_RGBf(0, 0, 51),      RGBu_TO_RGBf(255, 255, 255),
};

#endif // METAVISION_SDK_ANALYTICS_RANDOM_COLOR_MAP_H
