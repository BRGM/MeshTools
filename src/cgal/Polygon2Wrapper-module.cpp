#include "Polygon2Wrapper-module.h"

#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Polygon_2.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

#include "Epick_types.h"

typedef CGAL::Polygon_2<Kernel> CGAL_Polygon;
typedef CGAL_Polygon::Point_2 Point2;
typedef CGAL::Exact_predicates_tag Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, CGAL::Default, Itag>
    CDT;

static_assert(sizeof(Point2) == 2 * sizeof(double),
              "unconsistent sizes in memory");
static_assert(std::is_same<CDT::Point, Point2>::value,
              "unconsistent point types");

namespace py = pybind11;

struct Polygon {
  CGAL_Polygon boundary;
  std::vector<CGAL_Polygon> holes;
  Polygon(const Polygon&) = default;
  Polygon(Polygon&&) = default;
  Polygon(const CGAL_Polygon& border) : boundary{border} {}
  Polygon(CGAL_Polygon&& border)
      : boundary{std::forward<CGAL_Polygon>(border)} {}
  auto is_consistent() const {
    for (auto&& hole : holes) {
      if (!std::all_of(hole.vertices_begin(), hole.vertices_end(),
                       [this](const Point2& P) {
                         return this->boundary.has_on_bounded_side(P);
                       }))
        return false;
    }
    return true;
  }
  auto has_point_inside(const Point2& P) const {
    auto tmp =
        std::all_of(begin(holes), end(holes), [&P](const CGAL_Polygon& poly) {
          return poly.has_on_unbounded_side(P);
        });
    return boundary.has_on_bounded_side(P) &&
           std::all_of(begin(holes), end(holes),
                       [&P](const CGAL_Polygon& poly) {
                         return poly.has_on_unbounded_side(P);
                       });
  }
};

template <typename Iterator, typename Id = int>
auto rebind(Iterator p, Iterator pend) {
  std::map<CDT::Vertex_handle, Id> hmap;
  for (Id k = 0; p != pend; ++p, ++k) {
    hmap[p] = k;
  }
  return hmap;
}

auto mesh_polygon(const Polygon& polygon) {
  CDT cdt;
  auto insert_polygon_boundary_constraint = [&cdt](const CGAL_Polygon& P) {
    cdt.insert_constraint(P.vertices_begin(), P.vertices_end(), true);
  };
  insert_polygon_boundary_constraint(polygon.boundary);
  for (auto&& hole : polygon.holes) {
    insert_polygon_boundary_constraint(hole);
  }
  assert(cdt.is_valid());
  auto vmap = rebind(cdt.finite_vertices_begin(), cdt.finite_vertices_end());
  assert(vmap.size() == cdt.number_of_vertices());
  assert(vmap.size() ==
         std::distance(cdt.finite_vertices_begin(), cdt.finite_vertices_end()));
  auto vertices = py::array_t<double, py::array::c_style>{
      {(std::size_t)cdt.number_of_vertices(), (std::size_t)2}};
  auto p = reinterpret_cast<Point2*>(vertices.request().ptr);
  std::transform(cdt.finite_vertices_begin(), cdt.finite_vertices_end(), p,
                 [](const auto& v) { return v.point(); });
  auto triangles = py::array_t<int, py::array::c_style>{
      {(std::size_t)cdt.number_of_faces(), (std::size_t)3}};
  auto pv = reinterpret_cast<int*>(triangles.request().ptr);
  for (auto face = cdt.finite_faces_begin(); face != cdt.finite_faces_end();
       ++face) {
    for (int k = 0; k != 3; ++k, ++pv) {
      (*pv) = vmap[face->vertex(k)];
    }
  }
  auto in_polygon = py::array_t<bool, py::array::c_style>{
      static_cast<py::ssize_t>(cdt.number_of_faces())};
  auto p_in = reinterpret_cast<bool*>(in_polygon.request().ptr);
  for (auto face = cdt.finite_faces_begin(); face != cdt.finite_faces_end();
       ++face, ++p_in) {
    (*p_in) = polygon.has_point_inside(CGAL::centroid(cdt.triangle(face)));
  }
  return py::make_tuple(vertices, triangles, in_polygon);
}

struct Point2Buffer {
  Point2* data;
  std::size_t n;
  auto begin() noexcept { return data; }
  auto end() noexcept { return data + n; }
};

inline auto array_to_point_buffer(
    py::array_t<double, py::array::c_style>& points) {
  assert(points.ndim() == 2);
  assert(points.shape(1) == 2);
  static_assert(sizeof(Point2) == 2 * sizeof(double),
                "unconsistent sizes in memory");
  return Point2Buffer{reinterpret_cast<Point2*>(points.mutable_data(0, 0)),
                      static_cast<std::size_t>(points.shape(0))};
}

auto cgal_polygon_from_points_array(
    py::array_t<double, py::array::c_style>& points,
    const double squared_threshold) {
  assert(squared_threshold >= 0);
  auto buffer = array_to_point_buffer(points);
  auto p = buffer.data;
  auto n = buffer.n;
  assert(n > 2);
  // Remove the latest point if it is a duplicate of the first
  if (CGAL::squared_distance(*p, *(p + (n - 1))) < squared_threshold) {
    --n;
    assert(n > 2);
  }
  auto res = CGAL_Polygon{p, p + n};
  assert(res.is_simple());
  return res;
}

auto polygon_has_all_points_inside(
    const Polygon& polygon, py::array_t<double, py::array::c_style>& points) {
  auto buffer = array_to_point_buffer(points);
  return std::all_of(buffer.begin(), buffer.end(), [&polygon](const Point2& P) {
    return polygon.has_point_inside(P);
  });
}

auto polygon_has_points_inside(
    const Polygon& polygon, py::array_t<double, py::array::c_style>& points) {
  auto buffer = array_to_point_buffer(points);
  auto res =
      py::array_t<bool, py::array::c_style>{static_cast<py::ssize_t>(buffer.n)};
  auto pres = reinterpret_cast<bool*>(res.request().ptr);
  for (auto p = buffer.begin(); p != buffer.end(); ++p) {
    *pres = polygon.has_point_inside(*p);
    ++pres;
  }
  return res;
}

void add_polygon_wrapper(py::module& module) {
  typedef py::array_t<double, py::array::c_style> py_double_array;

  module.doc() = "pybind11 homemade CGAL Polygon2 interface";

  py::class_<Point2>(module, "Point2")
      .def(py::init<double, double>())
      .def_property_readonly(
          "coordinates",
          [](Point2& self) {
            return py_double_array{2, reinterpret_cast<double*>(&self)};
          })
      .def_property_readonly("x", &Point2::x)
      .def_property_readonly("y", &Point2::y)
      .def("__str__", [](Point2& self) {
        auto s = py::str("(%f,%f)");
        return s.format(self.x(), self.y());
      });

  py::class_<CGAL_Polygon>(module, "SimplePolygon")
      .def(
          py::init([](py_double_array& points, const double squared_threshold) {
            return std::make_unique<CGAL_Polygon>(
                cgal_polygon_from_points_array(points, squared_threshold));
          }),
          py::arg("points"), py::arg("squared_threshold") = 1e-10)
      .def("is_convex", &CGAL_Polygon::is_convex)
      .def_property_readonly("points",
                             [](const CGAL_Polygon& self) {
                               return py::make_iterator(self.vertices_begin(),
                                                        self.vertices_end());
                             })
      .def("has_point_inside", &CGAL_Polygon::has_on_bounded_side)
      .def("has_point_outside", &CGAL_Polygon::has_on_unbounded_side);

  py::class_<Polygon>(module, "Polygon")
      .def(py::init([](py::list polygons, const double squared_threshold) {
             assert(py::len(polygons) > 0);
             auto boundary = polygons[0].cast<py_double_array>();
             auto res = std::make_unique<Polygon>(
                 cgal_polygon_from_points_array(boundary, squared_threshold));
             for (std::size_t k = 1; k != py::len(polygons); ++k) {
               auto hole = polygons[k].cast<py_double_array>();
               res->holes.emplace_back(
                   cgal_polygon_from_points_array(hole, squared_threshold));
             }
             return res;
           }),
           py::arg("polygons"), py::arg("squared_threshold") = 1e-10)
      .def_readonly("boundary", &Polygon::boundary)
      .def_property_readonly("holes",
                             [](Polygon& self) {
                               return py::make_iterator(begin(self.holes),
                                                        end(self.holes));
                             })
      .def("has_holes", [](Polygon& self) { return !self.holes.empty(); })
      .def("nb_holes", [](Polygon& self) { return self.holes.size(); })
      .def("is_consistent", &Polygon::is_consistent)
      .def("has_point_inside", &Polygon::has_point_inside)
      .def("has_points_inside",
           [](const Polygon& self, py_double_array& points) {
             return polygon_has_points_inside(self, points);
           })
      .def("has_all_points_inside",
           [](const Polygon& self, py_double_array& points) {
             return polygon_has_all_points_inside(self, points);
           })
      .def("triangulate", &mesh_polygon);
}
