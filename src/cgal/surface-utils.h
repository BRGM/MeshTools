#pragma once

#include <CGAL/Polygon_mesh_processing/corefinement.h>

#include "Epick_types.h"

template <typename Constraint_map>
auto collect_constrained_edges_as_curves(const Triangulated_surface& S, const Constraint_map& constraints)
{

    typedef typename Triangulated_surface::Base Mesh;
    typedef typename Mesh::Vertex_index Vertex_index;
    typedef typename Mesh::Edge_index Edge_index;

    typedef std::list<Vertex_index> Curve;
    std::vector<Curve> curves;

    const Mesh& mesh = S;

    auto collect_constrained_edges = [](const Mesh& mesh, const auto& constraints) {
        std::vector<Edge_index> result;
        for (auto&& edge : mesh.edges()) {
            if (constraints[edge]) {
                result.emplace_back(edge);
            }
        }
        return result;
    };

    auto constrained_edges = collect_constrained_edges(mesh, constraints);

    auto edge_in_curve = std::map<Edge_index, bool>{};
    for (auto&& edge : constrained_edges) {
        edge_in_curve.emplace(edge, false);
    }
    auto find_free_edge = [&edge_in_curve]() {
        typedef boost::optional<Edge_index> Result;
        auto free_edge = std::find_if(
            begin(edge_in_curve), end(edge_in_curve),
            // find the first pair <key, value> with value = false, i.e. edge is not in curve
            [](auto&& pair) { return !pair.second; }
        );
        if (free_edge == end(edge_in_curve)) return Result{};
        return Result{ free_edge->first };
    };
    auto collect_vertices_until_corner = [&mesh, &edge_in_curve](Vertex_index start, auto out) {
        auto find_single_exit = [&edge_in_curve, &mesh](Vertex_index v) {
            //std::cerr << "looking for exit at " << mesh.point(v) << " "
            //    << std::count_if(begin(edge_in_curve), end(edge_in_curve), [](auto p) { return p.second; })
            //    << " edges in curves" << std::endl;
            typedef std::pair<Edge_index, Vertex_index> Exit;
            typedef boost::optional<Exit> Result;
            auto result = Result{};
            for (auto&& h : CGAL::halfedges_around_source(v, mesh)) {
                auto p = edge_in_curve.find(mesh.edge(h));
                if (p != end(edge_in_curve)) {
                    if (!p->second) { // edge is not on curve
                        if (result) { // already one exit found
                            return Result{};
                        }
                        else {
                            result = Exit{ p->first, mesh.target(h) };
                        }
                    }
                }
            }
            return result;
        };
        auto next_vertex_found = find_single_exit(start);
        while (next_vertex_found) {
            edge_in_curve[next_vertex_found->first] = true;
            start = next_vertex_found->second;
            *out = start;
            ++out;
            next_vertex_found = find_single_exit(start);
        }
    };
    for (
        auto edge_left = find_free_edge();
        edge_left;
        edge_left = find_free_edge()
        ) {
        auto starting_edge = *edge_left;
        curves.emplace_back();
        auto& curve = curves.back();
        edge_in_curve[starting_edge] = true;
        curve.emplace_back(mesh.vertex(starting_edge, 0));
        curve.emplace_back(mesh.vertex(starting_edge, 1));
        collect_vertices_until_corner(curve.back(), std::back_inserter(curve));
        collect_vertices_until_corner(curve.front(), std::front_inserter(curve));
    }
    assert(std::all_of(begin(edge_in_curve), end(edge_in_curve), [](auto&& p) {return p.second; }));

    return curves;

}

template <typename Output_iterator>
auto intersection_curves(Triangulated_surface& S1, Triangulated_surface& S2, Output_iterator out)
{

    typedef typename Triangulated_surface::Base Mesh;
    typedef Mesh::Vertex_index Vertex_index;
    typedef Mesh::Edge_index Edge_index;

    auto add_constraint_map = [](Mesh& mesh) {
        auto result = mesh.add_property_map<Edge_index, bool>("e:constrained", false);
        assert(result.second);
        return result.first;
    };
    auto constraints1 = add_constraint_map(S1);
    //auto constraints2 = add_constraint_map(S2);
    namespace parameters = CGAL::Polygon_mesh_processing::parameters;
    CGAL::Polygon_mesh_processing::corefine(
        static_cast<Mesh&>(S1), static_cast<Mesh&>(S2),
        parameters::edge_is_constrained_map(constraints1),
        //parameters::edge_is_constrained_map(constraints2),
        true
    );
    auto curves1 = collect_constrained_edges_as_curves(S1, constraints1);
    //auto curves2 = collect_curves(S2, constraints2);
    S1.remove_property_map(constraints1);
    //S2.remove_property_map(constraints2);
    for (auto&& curve : curves1) {
        Polyline polyline;
        polyline.reserve(curve.size());
        for (auto&& v1 : curve) {
            assert(S1.is_valid(v1));
            polyline.emplace_back(S1.point(v1));
        }
        (*out) = polyline;
        ++out;
    }

}

