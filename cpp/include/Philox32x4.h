#pragma once 

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <vector>

#include "GilesErfinv.h"

constexpr uint32_t NUM_ROUNDS = 10;
constexpr double kSqrt2 = 1.4142135623730950488;

inline std::array<uint32_t, 4> philox32x4(
    std::array<uint32_t, 4> counter,
    std::array<uint32_t, 2> key
)
{
        constexpr uint32_t M0 = 0xD2511F53;
        constexpr uint32_t M1 = 0xCD9E8D57;
        constexpr uint32_t W0 = 0x9E3779B9;
        constexpr uint32_t W1 = 0xBB67AE85;
         
        for(uint32_t i = 0; i < NUM_ROUNDS; ++i)
        {
            uint64_t prod0 = static_cast<uint64_t>(counter[0])*M0;
            uint64_t prod1 = static_cast<uint64_t>(counter[2])*M1;

            counter[0] = static_cast<uint32_t>(prod1 >> 32)^ counter[1] ^ key[0]; //hi xor r0 xor k0
            counter[1] = static_cast<uint32_t>(prod1); //lo
            counter[2] = static_cast<uint32_t>(prod0 >> 32)^ counter[3] ^ key[1]; //hi xor r1 xor k1
            counter[3] = static_cast<uint32_t>(prod0); //lo
            key[0] += W0;
            key[1] += W1;
        }

        return counter;

}

// Given argument num_rands, this method returns a
// vector of num_rands Philox32 random numbers.
inline std::vector<uint32_t> philox32x4_batch(uint32_t num_rands,
                                        uint32_t counter0_offset = 0,
                                        uint32_t counter1_offset = 0)
{
        std::vector<uint32_t> output_rands(num_rands); //TODO - avoid allocating a vector here

        uint32_t num_loops = num_rands/4;

        for (uint32_t loop = 0; loop < num_loops; ++loop)
        {
            auto out = philox32x4({counter0_offset + loop, counter1_offset + 0, 0, 0}, {0, 0});

            output_rands[loop*4] = out[0];
            output_rands[loop*4+1] = out[1];
            output_rands[loop*4+2] = out[2];
            output_rands[loop*4+3] = out[3];
        }

        if(num_rands % 4 != 0)
        {
            uint32_t remaining = num_rands % 4;
            auto out = philox32x4({counter0_offset + num_loops, counter1_offset + 0, 0, 0}, {0, 0});
            for(uint32_t i = 0; i < remaining; ++i)
            {
                output_rands[4*num_loops + i] = out[i];
            }
        }

        return output_rands;
}

// Given argument num_rands, this method returns a
// vector of num_rands sampled from normal distribution
template<typename T> std::vector<T> philox32x4_normal_batch(uint32_t num_rands,
                                                            uint32_t counter0_offset = 0,
                                                            uint32_t counter1_offset = 0);

// Given argument num_rands, this method returns a
// vector of num_rands floats sampled from normal distribution
template<>
inline std::vector<float> philox32x4_normal_batch<float>(uint32_t num_rands,
                                                            uint32_t counter0_offset,
                                                            uint32_t counter1_offset)
{
        std::vector<float> output_rands(num_rands);

        uint32_t num_loops = num_rands/4;

        auto to_normal = [](uint32_t arg){
            int32_t t = static_cast<int32_t>(2*(arg>>8) + 1) - (1 << 24);
            return static_cast<float>(kSqrt2)*erfinv_giles<float>(static_cast<float>(t) * 0x1.0p-24f);
        };

        for (uint32_t loop = 0; loop < num_loops; ++loop)
        {
            auto out = philox32x4({counter0_offset + loop, counter1_offset + 0, 0, 0}, {0, 0});

            for(uint32_t i = 0; i < 4; ++i)
                output_rands[4*loop + i] = to_normal(out[i]);
        }

        if(num_rands % 4 != 0)
        {
            uint32_t remaining = num_rands % 4;
            auto out = philox32x4({counter0_offset + num_loops, counter1_offset + 0, 0, 0}, {0, 0});
            for(uint32_t i = 0; i < remaining; ++i)
                output_rands[4*num_loops + i] = to_normal(out[i]);
        }

        return output_rands;
}

// Given argument num_rands, this method returns a
// vector of num_rands doubles sampled from normal distribution
template<>
inline std::vector<double> philox32x4_normal_batch<double>(uint32_t num_rands,
                                                            uint32_t counter0_offset,
                                                            uint32_t counter1_offset)
{
        std::vector<double> output_rands(num_rands);

        uint32_t num_loops = num_rands/2;

        auto to_normal = [](uint32_t a, uint32_t b){
            uint64_t combined = (static_cast<uint64_t>(a) << 32) | b;
            int64_t t = static_cast<int64_t>(2*(combined >> 11) + 1) - (int64_t(1) << 53);
            return kSqrt2*erfinv_giles<double>(static_cast<double>(t) * 0x1.0p-53);
        };

        for (uint32_t loop = 0; loop < num_loops; ++loop)
        {
            auto out = philox32x4({counter0_offset + loop, counter1_offset + 0, 0, 0}, {0, 0});
            for(int i = 0; i < 2; i++)
                output_rands[loop*2 + i] = to_normal(out[2*i], out[2*i + 1]);
        }

        // calculate the last double if num_rands%2 != 0
        if(num_rands % 2 != 0)
        {
            auto out = philox32x4({counter0_offset + num_loops, counter1_offset + 0, 0, 0}, {0, 0});
            output_rands[num_rands - 1] = to_normal(out[0], out[1]);
        }

        return output_rands;
}

template<typename W>
inline void philox32x4_mt_impl(uint32_t num_threads, uint32_t num_rands, W& work)
{
        if(num_threads == 0)
            throw std::invalid_argument("num_threads must be positive");

        std::vector<std::thread> threads(num_threads);

        for(uint32_t th_idx = 0; th_idx < num_threads; ++th_idx)
        {
            auto chunk_begin = static_cast<uint32_t>(uint64_t(th_idx) * num_rands / num_threads);
            auto chunk_end   = static_cast<uint32_t>(uint64_t(th_idx + 1) * num_rands / num_threads);

            if(chunk_begin < chunk_end) //check we are not in a situation where num_threads > num_rands
                threads[th_idx] = std::thread(work, 0, th_idx, chunk_begin, chunk_end);
        }

        for(auto& th: threads)
        {
            if(th.joinable())
                th.join();
        }
}

// Multithreaded implementation - given argument num_rands, this method returns a
// vector of num_rands Philox32 random numbers generated by num_threads threads
inline std::vector<uint32_t> philox32x4_batch_mt(uint32_t num_rands, uint32_t num_threads)
{
        std::vector<uint32_t> output_rands(num_rands);

        auto work = [&output_rands](uint32_t c0_offset, uint32_t c1_offset, uint32_t chunk_begin, uint32_t chunk_end)
        {
            auto chunk_len = chunk_end - chunk_begin;
            auto out = philox32x4_batch(chunk_len, c0_offset, c1_offset);
            for(uint32_t i = chunk_begin; i < chunk_end; ++i)
                output_rands[i] = out[i - chunk_begin];
        };

        philox32x4_mt_impl(num_threads, num_rands, work);

        return output_rands;
}



// Multithreaded implementation - given argument num_rands, this method returns a
// vector of num_rands float/double normal random numbers generated with Philox32x4. The
// work is done by num_threads threads.
template<typename T>
inline std::vector<T> philox32x4_normal_batch_mt(uint32_t num_rands, uint32_t num_threads)
{
        std::vector<T> output_rands(num_rands);

        auto work = [&output_rands](uint32_t c0_offset, uint32_t c1_offset, uint32_t chunk_begin, uint32_t chunk_end)
        {
            auto chunk_len = chunk_end - chunk_begin;
            auto out = philox32x4_normal_batch<T>(chunk_len, c0_offset, c1_offset);
            for(uint32_t i = chunk_begin; i < chunk_end; ++i)
                output_rands[i] = out[i - chunk_begin];
        };

        philox32x4_mt_impl(num_threads, num_rands, work);

        return output_rands;
}
