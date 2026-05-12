#include "utilities.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <iostream>
#include <numeric>
#include <cassert>
#include <iomanip>
#include <vector>
#include <deque>
#include <complex>
#include <algorithm>
#include <functional>
#include <ranges>
#include <execution>
#include <omp.h>
#include <xsimd/xsimd.hpp>
#pragma once

using namespace std;

template <typename R = array<char, 1>, typename L = function<void(void)>>
class SIMD_reduce_view
{

private:
    static array<char, 1> p_one;
    stringstream p_log;
    R &p_loop_state;
    L p_loop_action;

public:
    SIMD_reduce_view(R &loop_state = p_one, L loop_action = []() {}) : p_loop_state(loop_state), p_loop_action(loop_action) { p_log << setprecision(2); }
    string get_log() { return p_log.str(); }

    using Index = int;
    using Int = int32_t;
    using Real = float;
    using CReal = complex<Real>;
    template <typename T>
    using Container = vector<T, allocator<T>>; // TODO use no-init-allocator
    template <typename T>
    using AlignedContainer = std::vector<T, xsimd::default_allocator<T>>;
    static constexpr auto stExec = execution::unseq;     // single-threaded execution policy
    static constexpr auto mtExec = execution::par_unseq; // multi-threaded execution policy

    static constexpr Index default_n = 3;
    static constexpr Index default_m = 2;
    static constexpr Index default_N = 24;
    static constexpr Index default_Nout = 10;

    auto benchReduceIterator(Index N = default_N)
    {
        // Do not change
        Real sum;
        Index Nout = min(N, default_Nout);
        auto container = std::views::repeat(Real{10}, N);

        for (auto _ : p_loop_state)
        {
            sum = 0;
            for (auto it = container.begin(); it != container.end(); ++it) {
                sum += *it;
            }
        }
        p_log << "Iterator \\t" << sum << '\\n';
        return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceRange(Index N = default_N)
    {
        // Do not change
        Real sum;
        Index Nout = min(N, default_Nout);
        auto container = std::views::repeat(Real{10}, N);

        for (auto _ : p_loop_state)
        {
            sum = 0;
            for (const auto elem : container) {
                sum += elem;
            }
        }
        p_log << "Range \\t" << sum << '\\n';
        return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceStl(Index N = default_N)
    {
        // Do not change
        Real sum;
        Index Nout = min(N, default_Nout);
        auto container = std::views::repeat(Real{10}, N);

        for (auto _ : p_loop_state)
        {
            std::reduce(container.begin(), container.end(), Real{0});
        }
        p_log << "Stl \\t" << sum << '\\n';
        return tuple{N * sizeof(Real), 0};
    }
    auto benchReduceSimdStl(Index N = default_N)
    {
        // Do not change
        Real sum;
        Index Nout = min(N, default_Nout);
        auto container = std::views::repeat(Real{10}, N);

        for (auto _ : p_loop_state)
        {
            std::reduce(stExec, container.begin(), container.end(), Real{0});
        }
        p_log << "SimdStl \\t" << sum << '\\n';
        return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceSimdOmpIterator(Index N = default_N)
    {
        // Do not change
        Real sum;
        Index Nout = min(N, default_Nout);
        auto container = std::views::repeat(Real{10}, N);

        for (auto _ : p_loop_state)
        {
            sum = 0;
            #pragma omp simd reduction(+ : sum)
            for (auto it = container.begin(); it != container.end(); ++it) {
                sum += *it;
            }
        }
        p_log << "OmpIterator \\t" << sum << '\\n';
        return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceSimdOmpRange(Index N = default_N)
    {
        // Do not change
        Real sum;
        Index Nout = min(N, default_Nout);
        auto container = std::views::repeat(Real{10}, N);

        for (auto _ : p_loop_state)
        {
            sum = 0;
            #pragma omp simd reduction(+ : sum)
            for (const auto elem : container) {
                sum += elem;
            }
        }
        p_log << "OmpRange \\t" << sum << '\\n';
        return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceSimdXHorizontal(Index N = default_N)
    {
        // Do not change
        using batch = xsimd::batch<Real>;
        constexpr auto simd_width = batch::size;
        Real sum;
        Index Nout = min(N, default_Nout);
        std::array<Real, 3*simd_width> arr;
        std::fill(arr.begin(), arr.end(), 10);
        auto crosses_cache_line = [simd_width](std::uintptr_t address) {
            // address / 64 calculates the cache line index
            // -> then simply check if address + simd_width
            // has the same index
            return address / 64 != (address + simd_width - 1) / 64;
        };
        auto* ptr = arr.data();
        while (!crosses_cache_line(reinterpret_cast<std::uintptr_t>(ptr))) {
            ++ptr;
        }

        for (auto _ : p_loop_state)
        {
            sum = 0;
            for (int i{0}; i < N/simd_width; ++i) {
                batch simd_vec = batch::load_unaligned(ptr);
                sum += xsimd::reduce_add(simd_vec);
            }

            // Tail
            const int remainder = N % simd_width;
            for (int i{0}; i < remainder; ++i) {
                sum += *ptr;
            }
        }
        p_log << "xsimd_Horizontal \\t" << sum << '\\n';
        return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceSimdXVertical(Index N = default_N)
    {
        // Do not change
        using batch = xsimd::batch<Real>;
        constexpr auto simd_width = batch::size;
        Real sum;
        Index Nout = min(N, default_Nout);
        batch sum_vec;
        std::array<Real, 3*simd_width> arr;
        std::fill(arr.begin(), arr.end(), 10);
        auto crosses_cache_line = [simd_width](std::uintptr_t address) {
            // address / 64 calculates the cache line index
            // -> then simply check if address + simd_width
            // has the same index
            return address / 64 != (address + simd_width - 1) / 64;
        };
        auto* ptr = arr.data();
        while (!crosses_cache_line(reinterpret_cast<std::uintptr_t>(ptr))) {
            ++ptr;
        }

        for (auto _ : p_loop_state)
        {
            sum = 0;
            for (int i{0}; i < N/simd_width; ++i) {
                batch simd_vec = batch::load_unaligned(ptr);
                sum_vec += simd_vec;
            }
            sum += xsimd::reduce_add(sum_vec);

            // Tail
            const int remainder = N % simd_width;
            for (int i{0}; i < remainder; ++i) {
                sum += *ptr;
            }
        }

        p_log << "xsimd_Vertical\\t" << sum << '\\n';
        return tuple{N * sizeof(Real), 0};
    }

    auto benchReduceSimdXAlign(Index N = default_N)
    {
        // Do not change
        using batch = xsimd::batch<Real>;
        constexpr auto simd_width = batch::size;
        Real sum;
        Index Nout = min(N, default_Nout);
        alignas(64) std::array<Real, simd_width> arr;
        std::fill(arr.begin(), arr.end(), 10);

        for (auto _ : p_loop_state)
        {
            sum = 0;
            for (int i{0}; i < N/simd_width; ++i) {
                batch simd_vec = batch::load_aligned(arr.data());
                sum += xsimd::reduce_add(simd_vec);
            }

            // Tail
            const int remainder = N % simd_width;
            for (int i{0}; i < remainder; ++i) {
                sum += arr[0];  
            }
        }
        p_log << "xsimd_aligned \\t" << sum << '\\n';
        return tuple{N * sizeof(Real), 0};
    }
};

template <typename R, typename L>
array<char, 1> SIMD_reduce_view<R, L>::p_one = {0};

