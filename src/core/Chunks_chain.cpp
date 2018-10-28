#include "Chunks_chain.h"
#include "Variable_size_array_chunk.h"
#include "Composite_chunk.h"
#include "Chunks_chain_IO.h"

using namespace Chunks;

int main(int, const char**)
{

    std::cout << std::endl;
    std::cout << "Chain of double array..." << std::endl;
    using VSA = VariableSizeArrayChunk<double>;
    using ChainedVSA = Chunks_chain<VSA>;
    ChainedVSA chain;
    double pd[3] = { 3, 5, 7 };
    chain.emplace_back(3, pd);
    double pd2[5] = { 3, 5, 7, 28, 89 };
    chain.emplace_back(5, pd2);
    std::cout << chain << std::endl;

    std::cout << std::endl;
    std::cout << "Chain of composites..." << std::endl;
    using Composite = Composite_chunk<VSA>;
    using Chained_composites = Chunks_chain<Composite>;
    Chained_composites chain2;
    chain2.emplace_back(chain.begin(), chain.end());
    chain2.emplace_back(chain.begin(), chain.end());
    chain.clear();
    chain.emplace_back(&(pd2[0]), &(pd2[0]) + 4);
    chain2.emplace_back(chain.begin(), chain.end());
    std::cout << chain2 << std::endl;

}


