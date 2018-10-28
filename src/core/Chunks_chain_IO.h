#pragma once

#include <iostream>

#include "Chunks_chain.h"
#include "Variable_size_array_chunk.h"
#include "Composite_chunk.h"

namespace Chunks
{

    template <typename V>
    std::ostream& adapted_output(std::ostream& os, const V& v,
        const std::string& prefix, const std::string& separator, const std::string& suffix,
        const std::size_t& maxoutput)
    {
        auto p = v.begin();
        using value_type = decltype(*p);
        const std::size_t n = std::distance(p, v.end());
        os << prefix;
        if (n <= maxoutput) {
            std::copy(p, v.end(), std::ostream_iterator<value_type>(os, separator.c_str()));
        }
        else {
            std::advance(p, maxoutput / 2);
            std::copy(v.begin(), p, std::ostream_iterator<value_type>(os, separator.c_str()));
            os << separator << "..." << separator;
            std::advance(p, n - maxoutput);
            std::copy(p, v.cend(), std::ostream_iterator<value_type>(os, separator.c_str()));
        }
        os << suffix;
        return os;
    }

    template <typename T, typename SizeType>
    std::ostream& operator<<(std::ostream& os, const VariableSizeArrayChunk<T, SizeType>& vsa)
    {
        return Chunks::adapted_output(os, vsa, "(", ", ", ")", 6);
    }

    template <typename SubChunk, typename SizeType>
    std::ostream& operator<<(std::ostream& os, const Composite_chunk<SubChunk, SizeType>& composite)
    {
        return Chunks::adapted_output(os, composite, "Composite[", "; ", "]", 3);
    }

    template <typename Chunk, typename Word>
    std::ostream& operator<<(std::ostream& os, const Chunks_chain<Chunk, Word>& chain)
    {
        os << chain.nb_chunks() << " chained chunks:" << std::endl;
        return Chunks::adapted_output(os, chain, "[[\n", "\n", "]]\n", 6);
    }

} // namespace Chunks
