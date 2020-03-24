#include <vector>

#include "petrel-mesh.h"

int main(int argc, const char* argv[]) {
  auto vertices = std::vector<Point>{
      Point{0, 0}, Point{0, 0.2}, Point{0, 0.4}, Point{0, 0.6}, Point{0, 1},
      Point{1, 0}, Point{1, 0.6}, Point{1, 0.2}, Point{1, 1.}};
  typedef std::pair<int, int> Constraint;
  auto segments = std::vector<Constraint>{Constraint{0, 5}, Constraint{1, 6},
                                          Constraint{3, 8}, Constraint{0, 5},
                                          Constraint{2, 7}, Constraint{4, 8}};

  build_constrained_delaunay_triangulation(
      vertices.data(), &(segments.front().first), segments.size());
  return 0;
}
