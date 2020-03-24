# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:54:33 2018

@author: lopez
"""

import MeshTools.PetrelMesh as PM

filename = "precc_triangulation.txt"

cdt = PM.CDT(filename)

PM.connected_components(cdt)
