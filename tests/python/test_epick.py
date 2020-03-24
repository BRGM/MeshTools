# -*- coding: utf-8 -*-

import numpy as np
import MeshTools.CGALWrappers as CGAL


def test_epick():
    a = np.arange(9)
    try:
        pl = CGAL.Polyline(a)
    except CGAL.EpickException:
        pass
    a.shape = (-1, 3)
    pl = CGAL.Polyline(a)
    plv = pl.view()
    print(plv)
    plv *= 2
    for P in pl:
        print(P)
    plb = np.array(pl)
    print(plb)
    plb *= 2
    print(plb)


if __name__ == "__main__":
    test_epick()
