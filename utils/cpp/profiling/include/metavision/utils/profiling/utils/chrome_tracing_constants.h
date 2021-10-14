/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_UTILS_PROFILING_CHROME_TRACING_CONSTANTS_H
#define METAVISION_UTILS_PROFILING_CHROME_TRACING_CONSTANTS_H

#include <string>
#include <vector>

namespace Profiling {
namespace Constants {

static const char kDurationEventBeginType = 'B';
static const char kDurationEventEndType   = 'E';
static const char kCompleteEventType      = 'X';
static const char kInstantEventType       = 'i';
static const char kCounterEventType       = 'C';
static const char kGlobalScopeType        = 'g';
static const char kThreadScopeType        = 't';

static const std::string kColorArray[] = {
    "thread_state_uninterruptible", // 182, 125, 143
    "thread_state_iowait",          // 255, 140, 0
    "thread_state_running",         // 126, 200, 148
    "thread_state_runnable",        // 133, 160, 210
    "thread_state_sleeping",        // 240, 240, 240
    "thread_state_unknown",         // 199, 155, 125
    "background_memory_dump",       // 0, 180, 180
    "light_memory_dump",            // 0, 0, 180
    "detailed_memory_dump",         // 180, 0, 180
    "vsync_highlight_color",        // 0, 0, 255
    "generic_work",                 // 125, 125, 125
    "good",                         // 0, 125, 0
    "bad",                          // 180, 125, 0
    "terrible",                     // 180, 0, 0
    "black",                        // 0, 0, 0
    "grey",                         // 221, 221, 221
    "white",                        // 255, 255, 255
    "yellow",                       // 255, 255, 0
    "olive",                        // 100, 100, 0
    "rail_response",                // 67, 135, 253
    "rail_animation",               // 244, 74, 63
    "rail_idle",                    // 238, 142, 0
    "rail_load",                    // 13, 168, 97
    "startup",                      // 230, 230, 0
    "heap_dump_stack_frame",        // 128, 128, 128
    "heap_dump_child_node_arrow",   // 204, 102, 0
    "cq_build_running",             // 255, 255, 119
    "cq_build_passed",              // 153, 238, 102
    "cq_build_failed",              // 238, 136, 136
    "cq_build_abandoned",           // 187, 187, 187
    "cq_build_attempt_runnig",      // 222, 222, 75
    "cq_build_attempt_passed",      // 103, 218, 35
    "cq_build_attempt_failed"       // 197, 81, 8
};

enum Color : int {
    LightMauve = 0,  // 182, 125, 143
    Orange,          // 255, 140, 0
    SeafoamGreen,    // 126, 200, 148
    VistaBlue,       // 133, 160, 210
    WhiteSmoke,      // 240, 240, 240
    Tan,             // 199, 155, 125
    IrisBlue,        // 0, 180, 180
    MidnightBlue,    // 0, 0, 180
    DeepMagenta,     // 180, 0, 180
    Blue,            // 0, 0, 255
    Grey,            // 125, 125, 125
    Green,           // 0, 125, 0
    DarkGoldenrod,   // 180, 125, 0
    Peach,           // 180, 0, 0
    Black,           // 0, 0, 0
    LightGrey,       // 221, 221, 221
    White,           // 255, 255, 255
    Yellow,          // 255, 255, 0
    Olive,           // 100, 100, 0
    CornflowerBlue,  // 67, 135, 253
    SunsetOrange,    // 244, 74, 63
    Tangerine,       // 238, 142, 0
    ShamrockGreen,   // 13, 168, 97
    GreenishYellow,  // 230, 230, 0
    DarkGrey,        // 128, 128, 128
    Tawny,           // 204, 102, 0
    Lemon,           // 255, 255, 119
    Lime,            // 153, 238, 102
    Pink,            // 238, 136, 136
    Silver,          // 187, 187, 187
    ManzGreen,       // 222, 222, 75
    KellyGreen,      // 103, 218, 35
    FuzzyWuzzyBrown, // 197, 81, 8
    Undefined        // In this case, the color choice will be taken at random by the Chrome Tracing Viewer
};

} // namespace Constants
} // namespace Profiling

#endif // METAVISION_UTILS_PROFILING_CHROME_TRACING_CONSTANTS_H
