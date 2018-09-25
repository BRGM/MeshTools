#include <vector>
#include <stdexcept>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

#include <pybind11/numpy.h>

#include "EpickWrapper-module.h"
#include "Epick_types.h"
#include "mesh-pyutils.h"
#include "surface-utils.h"

namespace py = pybind11;

template <typename Target_type, typename Source_type = Target_type, int nbcols = 1>
struct XArrayWrapper
{
    XArrayWrapper() = delete;
    XArrayWrapper(const XArrayWrapper&) = delete;
    XArrayWrapper& operator=(const XArrayWrapper&) = delete;
    Target_type * data;
    std::size_t n;
    // CHECKME: is it usefull to move small structures like pointers and integers?
    XArrayWrapper(XArrayWrapper&& wrapper) :
        data{ wrapper.data },
        n{ wrapper.n }
    {}
    explicit XArrayWrapper(py::array_t<Source_type, py::array::c_style> a) :
        data{ nullptr },
        n{ 0 } {
        static_assert(sizeof(Target_type) == nbcols * sizeof(Source_type), "inconsistent sizes in memory");
        if (a.ndim() == 1) {
            if (nbcols == 1) {
                data = reinterpret_cast<Target_type*>(a.mutable_data(0));
            }
            else {
                throw EpickException("wrapper should have only one column");
            }
        }
        if (a.ndim() != 2) {
            throw EpickException("array dimension should be 2");
        }
        if (a.shape(1) != nbcols) {
            throw EpickException("inconsistent number of columns");
        }
        data = reinterpret_cast<Target_type*>(a.mutable_data(0, 0));
        n = a.shape(0);
    }
    auto begin() const { return data; }
    auto end() const { return data + n; }
    auto size() const { return n; }
};

template <typename Point_type>
using PointArrayWrapper = XArrayWrapper<Point_type, double, 3>;

template <int nbcols, typename Target_type, typename T>
inline auto array_wrapper(py::array_t<T, py::array::c_style> a) {
    return XArrayWrapper<Target_type, T, nbcols>{ a };
}

void add_epick_wrapper(py::module& module)
{

    module.doc() = "pybind11 homemade CGAL Epick Kernel wrapper";

    py::register_exception<EpickException>(module, "EpickException");

    py::class_<Point>(module, "Point")
        .def(py::init<double, double, double>())
        .def("__str__", [](const Point& self) {
        return py::str("Point({: f}, {: f}, {: f})").format(self.x(), self.y(), self.z());
    })
        ;

    py::class_<Vector>(module, "Vector")
        .def(py::init<double, double, double>())
        .def("__str__", [](const Point& self) {
        return py::str("Vector({: f}, {: f}, {: f})").format(self.x(), self.y(), self.z());
    })
        ;

    py::class_<Plane>(module, "Plane")
        .def(py::init<Point, Vector>())
        ;

    py::class_<Polyline>(module, "Polyline", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init([](py::array_t<double, py::array::c_style> points) {
        PointArrayWrapper<const Point> wrapper{ points };
        return std::make_unique<Polyline>(std::begin(wrapper), std::end(wrapper));
    }))
        .def("__iter__", [](const Polyline& self) {
        return py::make_iterator(begin(self), end(self));
    }, py::keep_alive<0, 1>())
        .def_buffer([](Polyline& self) -> py::buffer_info {
        return py::buffer_info(
            reinterpret_cast<double *>(self.data()),
            sizeof(double), py::format_descriptor<double>::format(),
            2,
            { self.size(), static_cast<std::size_t>(3) },
            { static_cast<std::size_t>(3) * sizeof(double), sizeof(double) }
        );
    })
        // FIXME: prefer the buffer_info approach?
        .def("view", [](Polyline& self) {
        static_assert(sizeof(Point) == 3 * sizeof(double), "Inconsistent sizes in memory.");
        typedef py::array_t<double, py::array::c_style> double_array;
        return double_array{
            {self.size(), static_cast<std::size_t>(3)},
            { static_cast<std::size_t>(3) * sizeof(double), sizeof(double) },
                reinterpret_cast<double *>(self.data()),
                double_array{} // used to create a handle so that the returned array is effectively a view
        };
    }, py::keep_alive<0, 1>())
        ;

    py::class_<Polylines>(module, "Polylines")
        .def(py::init<>())
        .def("__iter__", [](const Polylines& self) {
        return py::make_iterator(begin(self), end(self));
    }, py::keep_alive<0, 1>())
        ;

    py::class_<Triangulated_surface>(module, "TSurf")
        .def(py::init([](
            py::array_t<double, py::array::c_style> vertices,
            py::array_t<std::size_t, py::array::c_style> triangles
            ) {
        PointArrayWrapper<const Point> vertices_wrapper{ vertices };
        auto tsurf = std::make_unique<Triangulated_surface>();
        assert(tsurf);
        std::vector<Triangulated_surface::Vertex_index> vmap;
        vmap.reserve(vertices.shape(0));
        for (auto&& P : vertices_wrapper) {
            vmap.emplace_back(tsurf->add_vertex(P));
        }
        auto triangles_wrapper = array_wrapper<3, const std::array<std::size_t, 3>>(triangles);
        for (auto&& T : triangles_wrapper) {
            assert(vmap[T[0]] != tsurf->null_vertex());
            assert(vmap[T[1]] != tsurf->null_vertex());
            assert(vmap[T[2]] != tsurf->null_vertex());
#ifndef NDEBUG
            auto newface =
#endif
                tsurf->add_face(vmap[T[0]], vmap[T[1]], vmap[T[2]]);
            assert(new_face != tsurf->null_face());
        }
        return tsurf;
    }))
        .def("as_arrays", [](const Triangulated_surface& self) {
        return mesh_as_arrays(static_cast<const typename Triangulated_surface::Base&>(self));
    })
        ;

    module.def("intersection_curves", [](Triangulated_surface& S1, Triangulated_surface& S2) {
        auto polylines = std::make_unique<Polylines>();
        intersection_curves(S1, S2, std::back_inserter(*polylines));
        return polylines;
    });

}
