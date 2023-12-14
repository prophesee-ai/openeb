#ifndef WARNING_SUPRESSION_HPP
#define WARNING_SUPRESSION_HPP

#if defined(__GNUC__)

#define SUPRESS_DEPRECATION_WARNING(x)                                                             \
    _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"") \
        x _Pragma("GCC diagnostic pop")

#elif defined(__clang__)

#define SUPRESS_DEPRECATION_WARNING(x)                                                                 \
    _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"") \
        x _Pragma("clang diagnostic pop")

#elif defined(_MSC_VER)

// see reference to:
// https://learn.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warnings-c4200-through-c4399?view=msvc-170
#define SUPRESS_DEPRECATION_WARNING(x) \
    __pragma(warning(push)) __pragma(warning(disable : 4973 4974 4995 4996)) x __pragma(warning(pop))

#else

#define SUPRESS_DEPRECATION_WARNING(x) x

#endif

#endif // WARNING_SUPRESSION_HPP