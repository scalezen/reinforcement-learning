#pragma once 

#include <array>
#include <cstdint>
#include <iterator>
#include <thread>
#include <vector>

constexpr uint32_t NUM_ROUNDS = 10;

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
// vector of num_rands Philox32 random numbers
//
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
// vector of num_rands Philox32 random numbers
//
inline std::vector<uint32_t> philox32x4_batch_mt(uint32_t num_rands, uint32_t num_threads)
{
        uint32_t num_rands_per_thread = (num_rands + num_threads - 1)/num_threads;

        std::vector<std::thread> threads(num_threads);

        std::vector<uint32_t> output_rands(num_rands);

        auto work = [&output_rands](uint32_t num_rands, uint32_t c0_offset, uint32_t c1_offset, uint32_t chunk_begin, uint32_t chunk_end)
        {
            auto out = philox32x4_batch(num_rands, c0_offset, c1_offset);
            for(uint32_t i = chunk_begin; i < chunk_end; ++i)
                output_rands[i] = out[i - chunk_begin];
        };

        for(uint32_t th_idx = 0; th_idx < num_threads; ++th_idx)
        {
            auto chunk_begin = th_idx*num_rands_per_thread;
            auto chunk_end = std::min(num_rands, (th_idx+1)*num_rands_per_thread);
            if(chunk_begin < chunk_end) //check we are not in a situation where num_threads > num_rands
            {
                threads[th_idx] = std::thread(work, chunk_end-chunk_begin, 0, th_idx, chunk_begin, chunk_end);
            }
        }

        for(auto& th: threads)
        {
            if(th.joinable())
                th.join();
        }
        
        //collate output of all the threads in to output_rands
        return output_rands;
}
