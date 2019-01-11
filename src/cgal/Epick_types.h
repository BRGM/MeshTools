#pragma once

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

typedef CGAL::Epick Kernel;
typedef typename Kernel::Point_3 Point;
typedef typename Kernel::Vector_3 Vector;
typedef typename Kernel::Plane_3 Plane;

struct Polyline : std::vector<Point> {
    using std::vector<Point>::vector;
};

struct Polylines : std::list<Polyline> {
    using std::list<Polyline>::list;
};

using Triangulated_surface = CGAL::Surface_mesh<Point>;

struct EpickException : std::runtime_error
{
    explicit EpickException(const std::string& what) :
        std::runtime_error{ what }
    {}
};
