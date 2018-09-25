#include "EpickWrapper-module.h"
#include "DTMWrapper-module.h"
#include "C3t3Wrapper-module.h"

PYBIND11_MODULE(CGALWrappers, module)
{

    module.doc() = "pybind11 homemade quick and dirty CGAL wrappers";

    add_epick_wrapper(module);
    add_dtm_wrapper(module);
    add_c3t3_wrapper(module);

}
