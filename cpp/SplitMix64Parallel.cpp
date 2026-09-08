#include "SplitMix64.h"
#include "SplitMix64Parallel.h"



template <typename T>
void SplitMix64Parallel::run_parallel(size_t n, std::vector<T>& out) {
        if (n == 0) {
            return;
        }
        size_t base = n / n_threads_;
        size_t remainder = n % n_threads_;

        auto work = [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                out[i] = SplitMix64::mix(state_at(total_emitted_ + i));
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(n_threads_);

        size_t begin = 0;
        for (unsigned t = 0; t < n_threads_; ++t) {
            size_t chunk = base + (t < remainder ? 1 : 0);
            size_t end = begin + chunk;
            if (chunk > 0) {
                threads.emplace_back(work, begin, end);
            }
            begin = end;
        }
        for (auto &th : threads) {
            th.join();
        }
    }

std::vector<uint64_t> SplitMix64Parallel::next_batch(size_t n) {
        std::vector<uint64_t> out(n);
        run_parallel(n, out);
        total_emitted_ += n;
        return out;
    }

std::vector<double> SplitMix64Parallel::next_uniform(size_t n) {
        std::vector<double> out(n);
        run_parallel(n, out);
        total_emitted_ += n;
        return out;
    }