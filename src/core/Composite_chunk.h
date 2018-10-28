#pragma once

#include "Chunk.h"
#include "Base_chunks_chain.h"

namespace Chunks
{

    template <typename SubChunk, typename SizeType = std::size_t, typename WordType = unsigned char>
    struct Composite_chunk : Chunk
    {
    private:
        // redelete constructor and destructors
        // otherwise compilers may issue warning (cf. VS Studio Warning C4624)
        Composite_chunk() = delete;
        Composite_chunk(const Composite_chunk&) = delete;
        Composite_chunk& operator=(const Composite_chunk&) = delete;
        ~Composite_chunk() = delete;
    public:
        typedef SizeType size_type;
        typedef SubChunk value_type;
        typedef std::add_const_t<value_type> const_value_type;
    protected:
        typedef WordType word_type;
        typedef Base_chunks_chain<word_type> base_chain;
    public:
        typedef typename base_chain::template base_iterator<
            word_type *, value_type
        > iterator;
        typedef typename base_chain::template base_iterator<
            const word_type *, const_value_type
        > const_iterator;
    protected:
        size_type total_footprint;
        // this member is only declared as a short hand 
        // to take the adress of the first array element
        value_type first_subchunk;
    public:
        iterator begin() {
            // CHECKME: should this be alignof instead of sizeof?
            assert(reinterpret_cast<word_type *>(&(this->first_subchunk)) == reinterpret_cast<word_type *>(this) + sizeof(size_type));
            return iterator{
                reinterpret_cast<word_type *>(&(this->first_subchunk))
            };
        }
        iterator end() {
            return iterator{
                reinterpret_cast<word_type *>(this) + total_footprint
            };
        }
        const_iterator cbegin() const {
            // CHECKME: should this be alignof instead of sizeof?
            assert(reinterpret_cast<const word_type *>(&(this->first_subchunk)) == reinterpret_cast<const word_type *>(this) + sizeof(size_type));
            return const_iterator{
                reinterpret_cast<const word_type *>(&(this->first_subchunk))
            };
        }
        const_iterator cend() const {
            return const_iterator{
                reinterpret_cast<const word_type *>(this) + total_footprint
            };
        }
        auto begin() const { 
            return cbegin(); 
        }
        auto end() const { 
            return cend(); 
        }
        auto memory_footprint() const noexcept {
            return total_footprint;
        }
        template <typename Base_chunck_iterator>
        static constexpr auto memory_footprint(Base_chunck_iterator first, Base_chunck_iterator past_last) noexcept {
            static_assert(sizeof(word_type) == 1, "Chain words must have unit memory footprint.");
            auto mfp = sizeof(size_type);
            for (auto p = first; p != past_last; ++p) {
                mfp += p->memory_footprint();
            }
            return mfp;
        }
        // we do not use template parameter for the pointer
        // so that user must pay attention to the correct underlying type
        template <typename Base_chunck_iterator>
        static constexpr auto emplace(Composite_chunk *target, Base_chunck_iterator first, Base_chunck_iterator past_last) noexcept {
            auto mfp = sizeof(size_type);
            auto out = reinterpret_cast<word_type *>(&(target->first_subchunk));
            for (auto p = first; p != past_last; ++p) {
                const auto p_footprint = p->memory_footprint();
                mfp += p_footprint;
                auto cp = reinterpret_cast<word_type *>(&(*p));
                out = std::copy(cp, cp + p_footprint, out);
            }
            assert(mfp == memory_footprint(first, past_last));
            target->total_footprint = mfp;
        }
        value_type& operator[](const std::size_t i) {
            assert(i < nb_elements);
            return *(std::advance(begin(), i));
        }
        const_value_type& at(const std::size_t i) const {
            assert(i < nb_elements);
            return *(std::advance(begin(), i));
        }
        auto number_of_subchuncks() const {
            size_type n = 0;
            for (
                auto p = reinterpret_cast<const word_type*>(&first_subchunk);
                p != reinterpret_cast<const word_type*>(this) + total_footprint;
                p += reinterpret_cast<const SubChunk*>(p)->memory_footprint()
                ) {
                assert(p < reinterpret_cast<const word_type*>(this) + total_footprint);
                ++n;
            }
            return n;
        }
    };

}