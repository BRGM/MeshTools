#pragma once

#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_triangulation_face_base_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

#include <cassert>

struct Id_info {
  typedef int Id_type;
  static constexpr Id_type default_id = -1;
  Id_type id;
  Id_info() : id{default_id} {}
};

typedef CGAL::Epick Kernel;
typedef CGAL::Triangulation_vertex_base_with_info_2<Id_info, Kernel>
    Vertex_base;
typedef CGAL::Constrained_triangulation_face_base_2<Kernel> Face_base;
typedef CGAL::Triangulation_data_structure_2<Vertex_base, Face_base> Tds;
typedef CGAL::Exact_predicates_tag Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, Tds, Itag> CDT;
typedef CDT::Point Point;
typedef CDT::Segment Segment;
static_assert(Point::Ambient_dimension::value == 2, "wrong dimension");

auto build_constrained_delaunay_triangulation(const Point* const vertices,
                                              const int* const pairs,
                                              const std::size_t n) {
  assert(vertices != nullptr);
  assert(pairs != nullptr);
  CDT cdt;
  auto id = pairs;
  for (int k = 0; k < n; ++k) {
    assert(*id >= 0);
    auto v0 = cdt.insert(*(vertices + *id));
    v0->info().id = *id;
    ++id;
    assert(*id >= 0);
    auto v1 = cdt.insert(*(vertices + *id));
    v1->info().id = *id;
    cdt.insert_constraint(v0, v1);
    ++id;
  }
  return cdt;
}
