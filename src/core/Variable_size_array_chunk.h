#pragma once

#include "Chunk.h"

namespace Chunks
{

    template <typename T, typename SizeType = std::size_t, typename WordType = unsigned char>
    struct VariableSizeArrayChunk : Chunk
    {
    private:
        // redelete constructor and destructors
        // otherwise compilers may issue warning (cf. VS Studio Warning C4624)
        VariableSizeArrayChunk() = delete;
        VariableSizeArrayChunk(const VariableSizeArrayChunk&) = delete;
        VariableSizeArrayChunk& operator=(const VariableSizeArrayChunk&) = delete;
        ~VariableSizeArrayChunk() = delete;
    public:
        typedef SizeType size_type;
        typedef T value_type;
        typedef std::add_pointer_t<value_type> pointer;
        typedef std::add_const_t<value_type> const_value_type;
        typedef std::add_pointer_t<const_value_type> const_pointer;
    protected:
        typedef WordType word_type;
        size_type nb_elements;
        // this member is only declared as a short hand 
        // to take the adress of the first array element
        value_type first_element;
        static auto compute_memory_footprint(std::size_t n) noexcept {
            assert(n < std::numeric_limits<size_type>::max());
            return sizeof(size_type) + n * sizeof(value_type);
        }
    public:
        pointer begin() {
            // CHECKME: should this be alignof instead of sizeof?
            assert(&(this->first_element) == reinterpret_cast<pointer>(reinterpret_cast<word_type *>(this) + sizeof(size_type)));
            return &(this->first_element);
        }
        pointer end() {
            return begin() + nb_elements;
        }
        const_pointer cbegin() const {
            // CHECKME: should this be alignof instead of sizeof?
            assert(&(this->first_element) == reinterpret_cast<const_pointer>(reinterpret_cast<const word_type *>(this) + sizeof(size_type)));
            return &(this->first_element);
        }
        const_pointer cend() const {
            return cbegin() + nb_elements;
        }
        auto begin() const { return cbegin(); }
        auto end() const { return cend(); }
        auto memory_footprint() const noexcept {
            return compute_memory_footprint(nb_elements);
        }
        template <typename Input_iterator>
        static constexpr auto memory_footprint(Input_iterator first, Input_iterator past_last) noexcept {
            return compute_memory_footprint(
                std::distance(first, past_last)
            );
        }
        template <typename Input_iterator>
        static constexpr auto emplace(VariableSizeArrayChunk *target, Input_iterator first, Input_iterator past_last) noexcept {
            target->nb_elements = std::distance(first, past_last);
            std::copy(first, past_last, target->begin());
        }
        // we do not use template parameter for the pointer
        // so that user must pay attention to the correct underlying type
        static constexpr auto memory_footprint(std::size_t n, const_pointer) noexcept {
            return compute_memory_footprint(n);
        }
        static constexpr auto emplace(VariableSizeArrayChunk *target, std::size_t n, const_pointer source) noexcept {
            target->nb_elements = n;
            std::copy(source, source + n, target->begin());
        }
        value_type& operator[](const std::size_t i) {
            assert(i < nb_elements);
            return *(begin() + i);
        }
        const_value_type& at(const std::size_t i) const {
            assert(i < nb_elements);
            return *(begin() + i);
        }
        auto number_of_elements() const {
            return nb_elements;
        }
    };

} // namespace Chunks
