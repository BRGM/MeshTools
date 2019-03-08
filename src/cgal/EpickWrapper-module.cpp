#include <vector>
#include <stdexcept>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>

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

// CHECKME: does this avoid a copy?
template <typename T>
inline auto remove_forcecast(py::array_t<T, py::array::c_style | py::array::forcecast> a) {
    return py::array_t<T, py::array::c_style>{ a };
}

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
    //    .def("__getitem__", [](Polyline& self, std::size_t i) {
    //    if (i >= self.size()) {
    //        throw py::key_error{
    //            py::str{ "wrong index: {}" }.format(i)
    //        };
    //    }
    //    return self[i];
    //}, py::keep_alive<0, 1>())
        ;

    py::class_<Polylines>(module, "Polylines")
        .def(py::init<>())
        .def("__iter__", [](Polylines& self) {
        return py::make_iterator(begin(self), end(self));
    }, py::keep_alive<0, 1>())
    //    .def("__getitem__", [](Polylines& self, std::size_t i) {
    //    if (i >= self.size()) {
    //        throw py::key_error{ 
    //              py::str{"wrong index: {}"}.format(i) 
    //        };
    //    }
    //    return *(std::advance(self.begin(), i));
    //}, py::keep_alive<0, 1>())
        ;

    py::class_<Triangulated_surface>(module, "TSurf")
        .def(py::init<const Triangulated_surface&>()) // copy constructor
        .def(py::init([](
                py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
            py::array_t<std::size_t, py::array::c_style | py::array::forcecast> triangles
            ) {
        PointArrayWrapper<const Point> vertices_wrapper{ vertices };
        auto tsurf = std::make_unique<Triangulated_surface>();
        assert(tsurf);
        std::vector<Triangulated_surface::Vertex_index> vmap;
        vmap.reserve(vertices.shape(0));
        for (auto&& P : vertices_wrapper) {
            vmap.emplace_back(tsurf->add_vertex(P));
        }
        auto triangles_wrapper = array_wrapper<3, const std::array<std::size_t, 3>>(remove_forcecast(triangles));
        for (auto&& T : triangles_wrapper) {
            assert(vmap[T[0]] != tsurf->null_vertex());
            assert(vmap[T[1]] != tsurf->null_vertex());
            assert(vmap[T[2]] != tsurf->null_vertex());
#ifndef NDEBUG
            auto new_face =
#endif
                tsurf->add_face(vmap[T[0]], vmap[T[1]], vmap[T[2]]);
            assert(new_face != tsurf->null_face());
        }
        return tsurf;
    }))
        .def("as_arrays", [](const Triangulated_surface& self) {
        return mesh_as_arrays(self);
    })
        .def("submeshes", [](const Triangulated_surface& self) {
          std::map<Triangulated_surface::Face_index, Triangulated_surface::faces_size_type> facemap_container;
          boost::associative_property_map< std::map<Triangulated_surface::Face_index, Triangulated_surface::faces_size_type> > facemap(facemap_container);

          const size_t num_submeshes = CGAL::Polygon_mesh_processing::connected_components(self, facemap);
          py::list submeshes;

          if (num_submeshes == 1) { // No need to extract, skip the hard work.
            submeshes.append(self);
            return submeshes;
          }

          typedef typename Triangulated_surface::Face_index Face_index;
          typedef std::vector<Face_index> Mesh_part;
          std::vector<Mesh_part> parts{ num_submeshes };
          for (const auto& f : facemap_container) {
            parts[f.second].push_back(f.first);
          }
          
          for (const auto& part : parts) {
            submeshes.append(extract_submesh(self, begin(part), end(part)));
          }
          return submeshes;
    })
        .def("face_centers", [](const Triangulated_surface& self) {
        auto centers = py::array_t<double, py::array::c_style>{
            { static_cast<std::size_t>(self.number_of_faces()), static_cast<std::size_t>(3) }
        };
        static_assert(sizeof(Point) == 3 * sizeof(double), "inconsistent sizes in memory");
        auto pC = reinterpret_cast<Point *>(centers.mutable_data(0, 0));
        std::vector<Triangulated_surface::Vertex_index> fv;
        fv.reserve(3);
        for (auto&& f : self.faces()) {
            fv.clear();
            for (auto&& v : CGAL::vertices_around_face(self.halfedge(f), self)) {
                fv.emplace_back(v);
            }
            assert(fv.size() == 3);
            *pC = CGAL::centroid(self.point(fv[0]), self.point(fv[1]), self.point(fv[2]));
            ++pC;
        }
        return centers;
    })
        .def("to_off", [](const Triangulated_surface& self, py::str& filename) {
        auto os = std::ofstream{ filename };
        CGAL::write_off(os, self);
        os.close();
    })
        .def("remove_faces", [](Triangulated_surface& self, py::array_t<bool, py::array::c_style> where) {
        assert(where.ndim() == 1);
        assert(where.size() == self.number_of_faces());
        auto premove = where.data(0);
        for (auto&& f : self.faces()) {
            if (*premove) {
                CGAL::Euler::remove_face(self.halfedge(f), self);
            }
            ++premove;
        }
    })
        .def("is_closed", [](const Triangulated_surface& self) { return CGAL::is_closed(self); })
        .def_property_readonly("nb_vertices", &Triangulated_surface::number_of_vertices)
        .def("number_of_vertices", &Triangulated_surface::number_of_vertices)
        .def_property_readonly("nb_faces", &Triangulated_surface::number_of_faces)
        .def("number_of_faces", &Triangulated_surface::number_of_faces)
        .def("__str__", [](const Triangulated_surface& self) {
        return py::str{ "TSurf {} vertices {} faces" }.format(
            self.number_of_vertices(), self.number_of_faces()
        );
    })  
        .def("join", [](Triangulated_surface& self, const Triangulated_surface& other) {
        self.join(other);
    })
        .def("orient", [](Triangulated_surface& self, bool outward_orientation) {
        namespace parameters = CGAL::Polygon_mesh_processing::parameters;
        CGAL::Polygon_mesh_processing::orient(
            self, parameters::outward_orientation( outward_orientation )
        );
    }, py::arg("outward_orientation") = true)
        .def("self_intersections", [](const Triangulated_surface& self) {
        typedef typename Triangulated_surface::Face_index Face_index;
        typedef std::pair<Face_index, Face_index> Intersecting_faces;
        std::vector<Intersecting_faces> self_intersections;
        CGAL::Polygon_mesh_processing::self_intersections(
            self, std::back_inserter(self_intersections)
        );
        if (self_intersections.empty()) return py::object{ py::none{} };
        auto result = py::list{};
        for (auto&& faces : self_intersections) {
            result.append(
                py::make_tuple(
                    static_cast<std::size_t>(faces.first), 
                    static_cast<std::size_t>(faces.second)
                )
            );
        }
        return py::object{ result };
        })
        ;

    module.def("corefine", [](Triangulated_surface& S1, Triangulated_surface& S2) {
        const auto nv = S1.number_of_vertices();
        CGAL::Polygon_mesh_processing::corefine(S1, S2, true);
    });

    module.def("intersection_curves", [](Triangulated_surface& S1, Triangulated_surface& S2) {
        auto polylines = std::make_unique<Polylines>();
        intersection_curves(S1, S2, std::back_inserter(*polylines));
        return polylines;
    });

    module.def("fix_border_edges", [](Triangulated_surface& S) {
        CGAL::Polygon_mesh_processing::stitch_borders(S);
        for (const auto& he : S.halfedges()) {
            if (S.is_border(he)) {
                std::vector<Triangulated_surface::Face_index> patch_faces;
                std::vector<Triangulated_surface::Vertex_index> patch_vertices;
                CGAL::Polygon_mesh_processing::triangulate_and_refine_hole(S, he, std::back_inserter(patch_faces), std::back_inserter(patch_vertices));
            }
        }
    });

}
