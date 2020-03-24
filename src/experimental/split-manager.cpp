#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <utility>
#include <vector>

namespace py = pybind11;

typedef unsigned int Id_type;

template <typename IdType = Id_type>
struct Edge : std::pair<IdType, IdType> {
  typedef IdType id_type;
  typedef std::pair<IdType, IdType> base_type;
  Edge() = delete;
  Edge(IdType v1, IdType v2) : base_type{v1, v2} {
    if (this->first > this->second) {
      std::swap(this->first, this->second);
    }
  }
};

template <typename IdType = Id_type>
struct Faces_map {
  typedef IdType id_type;
  typedef Edge<IdType> edge_type;
  typedef std::vector<IdType> Face_nodes;
  typedef std::vector<IdType> Edge_faces;
  std::vector<Face_nodes> face_nodes;
  std::vector<edge_type> face_edges;
  std::map<edge_type, Edge_faces> edge_faces;
  void _collect_face_nodes(py::sequence nodes) {
    face_nodes.emplace_back();
    auto& nodes_copy = face_nodes.back();
    for (auto&& node : nodes) {
      nodes_copy.emplace_back(node.cast<IdType>());
    }
  }
  void _collect_edges(const IdType face_id, py::sequence nodes) {
    const auto n = static_cast<int>(py::len(nodes));
    for (int k = 0; k < n; ++k) {
      const auto e =
          edge_type{nodes[k - 1].cast<IdType>(), nodes[k].cast<IdType>()};
      auto p = edge_faces.find(e);
      if (p == end(edge_faces)) {
        auto result = edge_faces.emplace(e, Edge_faces{});
        assert(result.second);
        p = result.first;
      }
      p->second.push_back(face_id);
    }
  }
  IdType add_face(py::sequence nodes) {
    auto face_id = static_cast<IdType>(face_nodes.size());
    //_collect_face_nodes(nodes);
    _collect_edges(face_id, nodes);
    return face_id;
  }
  void register_faces(py::sequence faces) {
    for (auto&& face : faces) {
      add_face(face.cast<py::sequence>());
    }
  }
  void clear() {
    face_nodes.clear();
    edge_faces.clear();
  }
};

struct Splittables {
  typedef std::vector<std::size_t> Ids_base;
  struct Ids : Ids_base {
    using Ids_base::Ids_base;
    bool is_elementary() const { return empty(); }
  };

 private:
  std::vector<Ids> elements;

 public:
  Splittables(const std::size_t n) : elements{n} {
    assert(elements.size() == n);
    // for (std::size_t k = 0; k != n; ++k) {
    //    elements[k].push_back(k);
    //}
  }
  // bool is_splittable(const std::size_t k) const {
  //    assert(k < elements.size());
  //    assert(!elements[k].empty());
  //    return elements[k].size() == 1;
  //}
  Ids elementary_elements(const std::size_t k) const {
    assert(k < elements.size());
    Ids atoms;
    for (const auto& i : elements[k]) {
      if (elements[i].is_elementary()) {
        atoms.push_back(i);
      } else {
        auto tmp = elementary_elements(i);
        atoms.insert(end(atoms), begin(tmp), end(tmp));
      }
    }
    if (atoms.empty()) {
      assert(elements[k].is_elementary());
      atoms.push_back(k);
    }
    return atoms;
  }
  void split(std::size_t k, py::iterable l) {
    assert(k < elements.size());
    assert(elements[k].is_elementary());
    auto& ids = elements[k];
    for (auto& obj : l) {
      auto i = obj.cast<std::size_t>();
      assert(i != k);
      assert(elements[i].is_elementary());
      ids.push_back(i);
    }
  }
  auto __iter__() const {
    return py::make_iterator(begin(elements), end(elements));
  }
};

template <typename IdType>
py::str __repr__(const Edge<IdType>& edge) {
  return py::str{"Edge({:d},{:d})"}.format(edge.first, edge.second);
}

PYBIND11_MODULE(SplitManager, module) {
  module.doc() =
      "pybind11 split manager to help managing splitting elements (quick and "
      "dirty!!!)";

  py::class_<Splittables::Ids>(module, "Ids")
      .def_property_readonly("is_elementary", &Splittables::Ids::is_elementary)
      .def_property_readonly("array", [](const Splittables::Ids& self) {
        return py::array_t<std::size_t, py::array::c_style>{self.size(),
                                                            self.data()};
      });

  py::class_<Splittables>(module, "Splittables")
      .def(py::init<std::size_t>())
      .def("elementary_elements", &Splittables::elementary_elements)
      .def("split", &Splittables::split)
      .def("__iter__", &Splittables::__iter__);

  typedef Edge<Id_type> pyEdge;
  py::class_<pyEdge>(module, "Edge")
      .def(py::init<Id_type, Id_type>())
      .def(py::init([](py::sequence seq) {
        if (py::len(seq) != 2)
          throw std::runtime_error("edge only have two vertices");
        return std::make_unique<pyEdge>(seq[0].cast<Id_type>(),
                                        seq[1].cast<Id_type>());
      }))
      .def("__lt__",
           [](const pyEdge& self, const pyEdge& other) { return self < other; })
      .def("__repr__", &__repr__<Id_type>);

  typedef Faces_map<Id_type> pyFaces_map;
  typedef pyFaces_map::Face_nodes pyFace_nodes;

  py::class_<pyFace_nodes>(module, "Face_nodes")
      .def("__iter__", [](const pyFace_nodes& self) {
        return py::make_iterator(begin(self), end(self));
      });

  py::class_<pyFaces_map>(module, "Faces_map")
      .def(py::init<>())
      .def("clear", &pyFaces_map::clear)
      .def("add_face", &pyFaces_map::add_face)
      .def("register_faces", &pyFaces_map::register_faces)
      .def("faces",
           [](const pyFaces_map& self) {
             return py::make_iterator(begin(self.face_nodes),
                                      end(self.face_nodes));
           })
      .def("edge_faces", [](const pyFaces_map& self, const pyEdge& edge) {
        auto p = self.edge_faces.find(edge);
        if (p == end(self.edge_faces)) {
          throw py::key_error{py::str{"{} not found"}.format(__repr__(edge))};
        }
        return py::make_iterator(begin(p->second), end(p->second));
      });
}
