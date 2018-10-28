#pragma once

namespace Chunks
{

    template <typename Word = unsigned char>
    struct Base_chunks_chain
    {
        typedef Word word_type;
        // Chunk_type is passed as template to build const iterator
        template <typename BufferIterator, typename Chunk_type>
        class base_iterator : public std::iterator <
            std::input_iterator_tag,                             // iterator_category
            Chunk_type,                                          // value_type
            std::size_t,                                         // difference_type
            typename std::add_pointer<Chunk_type>::type,         // pointer
            typename std::add_lvalue_reference<Chunk_type>::type // reference
        > {
        private:
            BufferIterator position;
            typedef std::iterator <
                std::input_iterator_tag, Chunk_type, std::size_t,
                typename std::add_pointer<Chunk_type>::type,
                typename std::add_lvalue_reference<Chunk_type>::type
            > base;
        public:
            using typename base::pointer;
            using typename base::reference;
            explicit base_iterator(BufferIterator first_word_position) :
                position{ first_word_position } 
            {}
            pointer as_chunk_pointer() const {
                return reinterpret_cast<pointer>(&(*position));
            }
            reference as_chunk_reference() const {
                return *(as_chunk_pointer());
            }
            base_iterator& operator++() {
                static_assert(sizeof(word_type) == 1, "Chain words must have unit memory footprint.");
                assert(as_chunk_pointer() != nullptr);
                std::advance(position, as_chunk_pointer()->memory_footprint());
                return *this;
            }
            base_iterator operator++(int) {
                auto retval = *this;
                ++(*this);
                return retval;
            }
            bool operator==(const base_iterator& other) const {
                return position == other.position;
            }
            bool operator!=(const base_iterator& other) const {
                return !(*this == other);
            }
            reference operator*() const {
                return as_chunk_reference();
            }
            pointer operator->() const {
                return as_chunk_pointer();
            }
        };
    };

} // namespace Chunks
