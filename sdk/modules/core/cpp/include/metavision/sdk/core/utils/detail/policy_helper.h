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

#ifndef METAVISION_SDK_CORE_DETAIL_POLICY_HELPER_H
#define METAVISION_SDK_CORE_DETAIL_POLICY_HELPER_H

#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/push_front.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include "factory.h"
#include "factory_helper.h"

// clang-format off
/********************************************************************************
 * Implementation details of policies macro : this is private and may change    *
 ********************************************************************************/

// Those macros are used internally, don't use them directly
#define POLICY_FIRST_OF_EACH_(r, count, seq) (BOOST_PP_SEQ_ELEM(0, BOOST_PP_SEQ_ELEM(count, seq)))
#define POLICY_SECOND_OF_EACH_(r, count, seq) (BOOST_PP_SEQ_ELEM(1, BOOST_PP_SEQ_ELEM(count, seq)))
#define POLICY_TO_ENUM_(seq, macro) BOOST_PP_SEQ_ENUM(BOOST_PP_REPEAT(BOOST_PP_SEQ_SIZE(seq), macro, seq))

#define POLICY_BASE_(data) BOOST_PP_TUPLE_ELEM(2, 0, BOOST_PP_SEQ_HEAD(data))
#define POLICY_DERIVED_(data) BOOST_PP_TUPLE_ELEM(2, 1, BOOST_PP_SEQ_HEAD(data))

#define POLICY_CTOR_(data)                                                                                          \
    Metavision::detail::factory_helper::base_class_constructor<                                                                         \
        POLICY_BASE_(data),                                                                                         \
        POLICY_DERIVED_(data)<POLICY_BASE_(data), POLICY_TO_ENUM_(BOOST_PP_SEQ_TAIL(data), POLICY_SECOND_OF_EACH_)> \
    >()
#define POLICY_KEY_(data) std::make_tuple(POLICY_TO_ENUM_(BOOST_PP_SEQ_TAIL(data), POLICY_FIRST_OF_EACH_))
#define POLICY_TUPLE_TO_SEQ_(r, data, elem) (BOOST_PP_TUPLE_TO_SEQ(2, elem))
#define REGISTER_POLICY_(r, data) factory.register_object(POLICY_KEY_(data), POLICY_CTOR_(data));

/********************************************************************************
 * Macro calls to create policies and register them                             *
 ********************************************************************************/

// For compiler with variadic macro argument support, use POLICY and POLICIES to define policies parameters.
// Usage is more convenient since you don't have to provide the number of parameters when calling the macros.
// For older compilers, use POLICY_N and POLICIES_N.
#if BOOST_PP_VARIADICS == 1
/// @brief Defines all template parameters for a policy based class
/// Use as POLICY((v_1,c_1),(v_2,c_2),...,(v_n,c_n)) where c_i is the policy class used when the
/// corresponding i_th value is equal to v_i
#define POLICY(...) BOOST_PP_SEQ_FOR_EACH(POLICY_TUPLE_TO_SEQ_, ~, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

/// @brief Wraps a variadic amount of policies defined with the @ref POLICY macro call
/// Use as POLICIES(p_1,p_2,...,p_n) where p_i is the ith policy (defined by the @ref POLICY macro) of the
/// template class to be registered with @ref REGISTER_POLICIES
#define POLICIES(...) BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)
#else
// Use as POLICY_N(n, ((v_1,c_1),(v_2,c_2),...,(v_n,c_n))) where c_i is the policy class used when the
// corresponding i_th value is equal to v_i
#define POLICY_N(n, p) BOOST_PP_SEQ_FOR_EACH(POLICY_TUPLE_TO_SEQ_, ~, BOOST_PP_TUPLE_TO_SEQ(n, p))
// Use as POLICIES_N(n, (p_1,p_2,...,p_n)) where p_i is the ith policy (defined by the POLICY_N macro) of the
// template class to be registered with REGISTER_POLICIES
#define POLICIES_N(n, p) BOOST_PP_TUPLE_TO_SEQ(n, p)
#endif

/// @brief Use this macro to register all policies of a templated class to be used with your factory.
/// @param base Base class of the templated policy based class to be instantiated
/// @param derived Is the derived class to be instantiated
/// @param seq List of policies to register
/// Use it as REGISTER_POLICIES(base, derived, POLICIES(POLICY(...), ...))
/// derived must have the following form :
/// template <typename Base, Policy_1, Policy_2, ..., Policy_n> class Derived
/// where Base is the base class passed as the first argument of this macro call
/// and Policy_i is either a type or template parameter (i.e. typename or template <typename...> class)
/// and n is the number of policies to be registered.
/// Policies are registered with the POLICIES macro call
/// For an example usage, consult the file gray_events_generator_algorithm.cpp
#define REGISTER_POLICIES(base, derived, seq) \
    BOOST_PP_SEQ_FOR_EACH_PRODUCT_R(1, REGISTER_POLICY_, BOOST_PP_SEQ_PUSH_FRONT(seq, ((base, derived))))
// clang-format on

#endif // METAVISION_SDK_CORE_DETAIL_POLICY_HELPER_H
