import numpy as np

import MeshTools.CGALWrappers as CGAL

# fmt: off
border = np.array(
    [
        [ 0,  0],
        [ 1,  0],
        [ 1,  1],
        [ 0,  1],
    ],
    dtype=np.float64,
)
hole1 = np.array(
    [
        [ 0.2,  0.1],
        [ 0.5,  0.15],
        [ 0.35,  0.7],
    ],
    dtype=np.float64,
)
hole2 = np.array(
    [
        [ 0.6,  0.4],
        [ 0.9,  0.4],
        [ 0.9,  0.9],
        [ 0.5,  0.8],
    ],
    dtype=np.float64,
)
# fmt: on


def draw(polygon, seeds):

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    def draw_polygon(polygon, color):
        points = np.array([P.coordinates for P in polygon.points])
        points = np.vstack([points, points[0]])  # close polygon
        plt.plot(points[:, 0], points[:, 1], c=color)

    plt.clf()
    draw_polygon(polygon.boundary, "blue")
    for hole in polygon.holes:
        draw_polygon(hole, "red")
    plt.plot(seeds[:, 0], seeds[:, 1], "ok")
    plt.savefig("test_polygon.png")


def draw_triangles(vertices, triangles, inside):

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    plt.clf()
    plt.tripcolor(vertices[:, 0], vertices[:, 1], triangles, inside, shading="flat")
    plt.triplot(vertices[:, 0], vertices[:, 1], triangles)
    plt.savefig("test_polygon_triangles.png")


def test_polygon():
    polygon = CGAL.SimplePolygon(border)
    assert polygon.has_point_inside(CGAL.Point2(0.5, 0.5))
    polygon = CGAL.Polygon([border, hole1, hole2])
    assert polygon.is_consistent()
    nseeds = 1000
    points = np.random.random((nseeds, 2))
    keep = polygon.has_points_inside(points)
    print(f"{np.sum(keep)} points kept out of {nseeds}")
    vertices, triangles, inside = polygon.triangulate()
    draw(polygon, points[keep])
    draw_triangles(vertices, triangles, inside)


if __name__ == "__main__":
    test_polygon()
