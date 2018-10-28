#pragma once

#include <memory>
#include <vector>
#include <limits>
#include <cassert>
#include <iterator>
#include <algorithm>
#include <string>

#include "Chunk.h"
#include "Base_chunks_chain.h"

namespace Chunks
{

    // chunks must know their memory footprint which is different from sizeof(Chunk)
    // otherwise we could directly use a vector
    template <typename Chunk_type, typename Word = unsigned char>
    struct Chunks_chain: Base_chunks_chain<Word>
    {
        typedef Base_chunks_chain<Word> base;
        using typename base::word_type;
        typedef Chunk_type chunk_type;
    protected:
        typedef std::vector<word_type> buffer_type;
        buffer_type buffer;
    public:
        typedef typename base::template base_iterator<
            typename buffer_type::iterator,
            chunk_type> iterator;
        typedef typename base::template base_iterator<
            typename buffer_type::const_iterator,
            typename std::add_const<chunk_type>::type> const_iterator;
        Chunks_chain() {
            static_assert(std::is_base_of<Chunk, chunk_type>::value, "Chunks must derive from base Chunk.");
        }
        template <typename ...Ts>
        iterator emplace_back(Ts&&... args) {
            static_assert(sizeof...(Ts) > 0, "Chunck have no default constructors neither emplacement functions with no arguments.");
            static_assert(sizeof(word_type) == 1, "Chain words must have unit memory footprint.");
            const auto preallocation_size = buffer.size();
            const auto chunck_size = chunk_type::memory_footprint(std::forward<Ts>(args)...);
            buffer.resize(preallocation_size + chunck_size);
            auto emplaced = iterator{ buffer.begin() + preallocation_size };
            chunk_type::emplace(emplaced.as_chunk_pointer(), std::forward<Ts>(args)...);
            return emplaced;
        }
        void clear() {
            buffer.clear();
        }
        std::size_t nb_chunks() const noexcept {
            return std::distance(const_iterator{ buffer.cbegin() }, const_iterator{ buffer.cend() });
        }
        std::size_t nb_words() const noexcept {
            static_assert(sizeof(word_type) == 1, "Chain words must have unit memory footprint.");
            return buffer.size();
        }
        auto begin() {
            return iterator{ buffer.begin() };
        }
        auto end() {
            return iterator{ buffer.end() };
        }
        auto cbegin() const {
            return const_iterator{ buffer.cbegin() };
        }
        auto cend() const {
            return const_iterator{ buffer.cend() };
        }
        auto begin() const {
            return cbegin();
        }
        auto end() const {
            return cend();
        }
    };

} // namespace Chunks 

