# @Time    : 2018.12.04
# @Author  : kawa Yeung
# @Licence : bio-totem
# @Refence : http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python


import math

import numpy as np
import pylab as pl
import shapely.geometry as geometry
from descartes import PolygonPatch
from scipy.spatial import Delaunay
from shapely.ops import cascaded_union, polygonize


def plot_polygon(polygon):
    """
    Plot polygon
    :param polygon:
    :return:
    """

    fig = pl.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    margin = .3
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    # patch = PolygonPatch(polygon, fc='#999999', ec='#000000', fill=False, zorder=-1)
    patch = PolygonPatch(polygon, ec='red', fill=False, lw=1, zorder=-1)
    ax.add_patch(patch)

    return fig


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """

    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull.convex_hull, None

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    tri = Delaunay(points)
    edges = set()
    edge_points = []

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # Here's the radius filter.
        # print circum_r
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, points, ia, ib)
            add_edge(edges, edge_points, points, ib, ic)
            add_edge(edges, edge_points, points, ic, ia)

    if edge_points == []:
        raise Exception(f"alpha: {alpha} set too big")

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))

    return cascaded_union(triangles), edge_points


if __name__ == "__main__":
    point1 = np.array([
        [1, 1],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 3],
        [3, 4],
        [4, 3],
        [5, 2],
        [5, 1],
        [4, 1],
        [3, 1],
        [2, 1]
    ])

    point2 = np.array([
        [6, 6],
        [6, 7],
        [7, 7],
        [7, 6]
    ])

    # points = np.row_stack((point1, point2))
    points = np.array([
        [1, 1],
        [2, 1],
        [2, 2],
        [1, 2]
    ])

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    concave_hull, edge_points = alpha_shape(points, alpha=1.3)

    _ = plot_polygon(concave_hull)
    # _ = pl.plot(x,y,'o', color='#f16824')

    pl.show()
