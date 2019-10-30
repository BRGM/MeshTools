Hybrid mesh generation
======================

Let's suppose that you have vertices in file named ``NODES.txt`` with
vertices coordinates as follows:

.. code-block:: text

    1 -1697.056534461986 -1697.056015233381 0.0
    2 -1437.7840000186845 -1437.7835762743787 0.0
    3 -1697.056534461986 -1697.056015233381 200.0
    4 -1437.7840000186861 -1437.7835762743769 200.00001525873645
    ...

and a file named ``MIXED.txt`` with cell connectivity given 
with 1 based indexing :

.. code-block:: text

    1 3137 2615 2617 3138 2616 2618
    2 3137 2617 2619 3138 2618 2620
    3 3137 2619 2621 3138 2620 2622
    4 3137 2621 2623 3138 2622 2624
    5 3137 2623 2625 3138 2624 2626
    6 3137 2625 2627 3138 2626 2628
    ...

We have first to import the previous tables as lists:

.. literalinclude:: hybrid_mesh.py
   :language: python
   :lines: 6-23

Then convert vertices to numpy array and described the mesh elements
using **a zero-based** node indexing
Cells with 6 vertices are considered to be wedges en cells with
8 elements are considered to be Hexahedra.
Meshtools follows the legacy vtk cell format to describe a cell 
from its nodes (cf. `page 9 of the specification document <https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf#page=9>`_).

.. literalinclude:: hybrid_mesh.py
   :language: python
   :lines: 22-36

Then we can create the mesh object from the collected information:

.. literalinclude:: hybrid_mesh.py
   :language: python
   :lines: 38

This mesh object can be passed to ComPASS or exported to paraview vtu format.

.. literalinclude:: hybrid_mesh.py
   :language: python
   :lines: 40

Full script
-----------

.. literalinclude:: hybrid_mesh.py
   :language: python
