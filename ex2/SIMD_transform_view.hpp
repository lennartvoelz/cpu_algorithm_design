#include "utilities.hpp"
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
class SIMD_transform_view
{

private:
    static array<char, 1> p_one;
    stringstream p_log;
    R &p_loop_state;
    L p_loop_action;

public:
    SIMD_transform_view(R &loop_state = p_one, L loop_action = []() {}) : p_loop_state(loop_state), p_loop_action(loop_action) { p_log << setprecision(2); }
    string get_log() { return p_log.str(); }

    using Index = int;
    using Int = int32_t;
    using Real = float;
    using CReal = complex<Real>;
    template <typename T>
    using Container = vector<T, allocator<T>>;
    template <typename T>
    using AlignedContainer = std::vector<T, xsimd::default_allocator<T>>;
    static constexpr auto stExec = execution::unseq;     // single-threaded execution policy
    static constexpr auto mtExec = execution::par_unseq; // multi-threaded execution policy

    static constexpr Index default_n = 3;
    static constexpr Index default_m = 2;
    static constexpr Index default_N = 24;
    static constexpr Index default_Nout = 10;

    auto benchTransformIterator(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        auto V = std::views::iota(0, N) | std::views::transform([](int x) {
            return static_cast<Real>(x);
        });
        std::vector<Real> W(256);
        std::iota(W.begin(), W.end(), Real{2});
        auto mod256_v = std::views::iota(0, N) | std::views::transform([](int x) {
            return x % 256;
        });

        for (auto _ : p_loop_state)
        {
            auto w_idx = mod256_v.begin();
            for (auto v = V.begin(); v != V.end(); ++v, ++w_idx) {
                W[*w_idx] = a * (*v) + W[*w_idx]; 
            }
            p_loop_action();
        }

        p_log << "Iterator \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformIteratorInnerLoop(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        constexpr int simd_width = 8;
        auto V = std::views::iota(0, N) | std::views::transform([](int x) {
            return static_cast<Real>(x);
        });
        std::vector<Real> W(256);
        std::iota(W.begin(), W.end(), Real{2});
        auto mod256 = [](int low, int high) {
            return std::views::iota(low, high) | std::views::transform([](int x) {
                return x % 256;
            });            
        };

        for (auto _ : p_loop_state)
        {
            for (int block{0}; block < N; block += simd_width) {
                auto v_chunk = V | std::views::drop(block)
                                 | std::views::take(simd_width);
                
                auto w_idx = mod256(block, block + simd_width);       
                auto w = w_idx.begin();

                for (auto v = v_chunk.begin(); v != v_chunk.end(); ++v, ++w) {
                    W[*w] = a * (*v) + W[*w];
                }
            }
            p_loop_action();
        }

        p_log << "IteratorInnerLoop \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformRange(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        auto V = std::views::iota(0, N) | std::views::transform([](int x) {
            return static_cast<Real>(x);
        });
        std::vector<Real> W(256);
        std::iota(W.begin(), W.end(), Real{2});
        auto mod256_v = std::views::iota(0, N) | std::views::transform([](int x) {
            return x % 256;
        });

        for (auto _ : p_loop_state)
        {
            for (auto [v, idx] : std::views::zip(V, mod256_v)) {
                W[idx] = a * v + W[idx];
            }
            p_loop_action();
        }

        p_log << "Range \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformRangeInnerLoop(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        constexpr int simd_width = 8;
        auto V = std::views::iota(0, N) | std::views::transform([](int x) {
            return static_cast<Real>(x);
        });
        std::vector<Real> W(256);
        std::iota(W.begin(), W.end(), Real{2});
        auto mod256 = [](int low, int high) {
            return std::views::iota(low, high) | std::views::transform([](int x) {
                return x % 256;
            });            
        };

        for (auto _ : p_loop_state)
        {
            for (int block{0}; block < N; block += simd_width) {
                auto v_chunk = V | std::views::drop(block)
                                 | std::views::take(simd_width);
                
                auto w_idx = mod256(block, block + simd_width);       
                
                for (auto [v, w] : std::views::zip(v_chunk, w_idx)) {
                    W[w] = a * v + W[w];
                }
            }
            p_loop_action();
        }

        p_log << "RangeInnerLoop \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
    auto benchTransformStl(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        auto V = std::views::iota(0, N) | std::views::transform([](int x) {
            return static_cast<Real>(x);
        });
        std::vector<Real> W(256);
        std::iota(W.begin(), W.end(), Real{2});
        auto W_v = std::views::iota(0, N) | std::views::transform([&W](int x) -> Real& {
            return W[x%256]; 
        });

        for (auto _ : p_loop_state)
        {
            std::transform(V.begin(), V.end(), W_v.begin(), W_v.begin(),
                           [a](const auto v, const auto w) {
                              return a * v + w;
                           }
            );
            p_loop_action();
        }

        p_log << "Stl \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchTransformSimdStl(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        auto V = std::views::iota(0, N) | std::views::transform([](int x) {
            return static_cast<Real>(x);
        });
        std::vector<Real> W(256);
        std::iota(W.begin(), W.end(), Real{2});
        auto W_v = std::views::iota(0, N) | std::views::transform([&W](int x) -> Real& {
            return W[x%256]; 
        });

        for (auto _ : p_loop_state)
        {
            std::transform(stExec, V.begin(), V.end(), W_v.begin(), W_v.begin(),
                           [a](const auto v, const auto w) {
                              return a * v + w;
                           }
            );
            p_loop_action();
        }

        p_log << "SimdStl \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchXsimdTransform(Index N = default_N)
    {
        // Do not change
        using batch = xsimd::batch<Real>;
        constexpr auto simd_width = batch::size;
        Index Nout = min(N, default_Nout);
        auto V = std::views::iota(0, N) | std::views::transform([](int x) {
           return static_cast<Real>(x); 
        });
        std::vector<Real> W(256);
        std::iota(W.begin(), W.end(), Real{2});
        auto mod256 = [](int low, int high) {
            return std::views::iota(low, high) | std::views::transform([](int x) {
                return x % 256;
            });
        };

        for (auto _ : p_loop_state)
        {
            for (int block{0}; block < N; block += simd_width) {
                auto V_chunk = V | std::views::drop(block)
                                 | std::views::take(simd_width)
                                 | std::ranges::to<std::vector>();

                auto w_idx = mod256(block, block + simd_width);
                // This always works because index wrap around
                // is a multiple of the simd_width
                batch w_vec = batch::load_unaligned(W.data() + *w_idx.begin());
                batch v_vec = batch::load_unaligned(V_chunk.data());
                batch a_vec(-1.0f);
                w_vec = a_vec * v_vec + w_vec;
                w_vec.store_unaligned(W.data() + *w_idx.begin());
            }

            p_loop_action();
        }
        p_log << "xsimd \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchXsimdTransformAligned(Index N = default_N)
    {
        // Do not change
        using batch = xsimd::batch<Real>;
        constexpr auto simd_width = batch::size;
        using aligned_vec = std::vector<Real, xsimd::aligned_allocator<Real, alignof(batch)>>;
        Index Nout = min(N, default_Nout);
        auto V = std::views::iota(0, N) | std::views::transform([](int x) {
           return static_cast<Real>(x); 
        });
        aligned_vec W(256);
        std::iota(W.begin(), W.end(), Real{2});
        auto mod256 = [](int low, int high) {
            return std::views::iota(low, high) | std::views::transform([](int x) {
                return x % 256;
            });
        };

        for (auto _ : p_loop_state)
        {
            for (int block{0}; block < N; block += simd_width) {
                auto V_chunk = V | std::views::drop(block)
                                 | std::views::take(simd_width)
                                 | std::ranges::to<aligned_vec>();

                auto w_idx = mod256(block, block + simd_width);
                // This always works because index wrap around
                // is a multiple of the simd_width
                batch w_vec = batch::load_aligned(W.data() + *w_idx.begin());
                batch v_vec = batch::load_aligned(V_chunk.data());
                batch a_vec(-1.0f);
                w_vec = a_vec * v_vec + w_vec;
                w_vec.store_aligned(W.data() + *w_idx.begin());
            }

            p_loop_action();
        }

        p_log << "xsimd_aligned \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformIterator(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        std::vector<Real> W(256);
        std::iota(W.begin(), W.end(), Real{2});
        auto V = std::views::iota(0, N) | std::views::transform([](int x) {
            return static_cast<Real>(x);
        });
        auto mod256_v = std::views::iota(0, N) | std::views::transform([](int x) {
            return x % 256;
        });

        for (auto _ : p_loop_state)
        {
            auto w_idx = mod256_v.begin();
            #pragma omp simd 
            for (auto v = V.begin(); v != V.end(); ++v) {
                W[*w_idx] = a * (*v) + W[*w_idx];
                ++w_idx;
            }
            p_loop_action();
        }
        p_log << "OmpIterator \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
    auto benchOmpSimdTransformIteratorInnerLoop(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        std::vector<Real> W(256);
        std::iota(W.begin(), W.end(), Real{2});
        auto V = std::views::iota(0, N) | std::views::transform([](int x) {
            return static_cast<Real>(x);
        });
        auto mod256 = [](int low, int high) {
            return std::views::iota(low, high) | std::views::transform([](int x) {
                return x % 256;
            });            
        };

        constexpr int simd_width{8};

        for (auto _ : p_loop_state)
        {
            for (int block{0}; block < N; block += simd_width) {
                auto V_chunk = V | std::views::drop(block)
                            | std::views::take(simd_width);

                auto w_idx = mod256(block, block + simd_width);
                auto w = w_idx.begin();

                #pragma omp simd
                for (auto v = V_chunk.begin(); v != V_chunk.end(); ++v) {
                    W[*w] = a * (*v) + W[*w];
                    ++w;
                }
            }
            p_loop_action();
        }
        p_log << "OmpIteratorInnerLoop \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformRange(Index N = default_N)
    {
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        constexpr int simd_width = 8;
        auto V = std::views::iota(0, N) | std::views::transform([](int x) {
            return static_cast<Real>(x);
        });
        std::vector<Real> W(256);
        std::iota(W.begin(), W.end(), Real{2});
        auto mod256_v = std::views::iota(0, N) | std::views::transform([](int x) {
            return x % 256;
        });            

        for (auto _ : p_loop_state)
        {
            #pragma omp simd
            for (auto [v, w] : std::views::zip(V, mod256_v)) {
                W[w] = a * v + W[w];
            }
            p_loop_action();
        }
     
        p_log << "OmpRange \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }

    auto benchOmpSimdTransformRangeInnerLoop(Index N = default_N)
    {
        // Do not change
        Real a = -1.0f;
        Index Nout = min(N, default_Nout);
        constexpr int simd_width = 8;
        auto V = std::views::iota(0, N) | std::views::transform([](int x) {
            return static_cast<Real>(x);
        });
        std::vector<Real> W(256);
        std::iota(W.begin(), W.end(), Real{2});
        auto mod256 = [](int low, int high) {
            return std::views::iota(low, high) | std::views::transform([](int x) {
                return x % 256;
            });            
        };

        for (auto _ : p_loop_state)
        {
            for (int block{0}; block < N; block += simd_width) {
                auto v_chunk = V | std::views::drop(block)
                                 | std::views::take(simd_width);
                
                auto w_idx = mod256(block, block + simd_width);       
                #pragma omp simd 
                for (auto [v, w] : std::views::zip(v_chunk, w_idx)) {
                    W[w] = a * v + W[w];
                }
            }
            p_loop_action();
        }

        p_log << "OmpRangeInnerLoop \t" << views::take(W, Nout) << '\n';
        return tuple{N * sizeof(Real) * 2, N * sizeof(Real)};
    }
};
template <typename R, typename L>
array<char, 1> SIMD_transform_view<R, L>::p_one = {0};
