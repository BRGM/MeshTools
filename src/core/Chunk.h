#pragma once

namespace Chunks
{

    /** Chuncks are never constructed directly. */
    struct Chunk {
    private:
        // no constructors neither destructors: chunk can only be emplaced
        Chunk() = delete;
        Chunk(const Chunk&) = delete;
        Chunk& operator=(const Chunk&) = delete;
        ~Chunk() = delete;
    };

} // namespace Chunks
