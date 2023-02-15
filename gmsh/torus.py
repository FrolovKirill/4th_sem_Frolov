import gmsh
import sys

gmsh.initialize()

gmsh.model.add("torus")

lc = 1e-1

gmsh.model.geo.addPoint(0, 0, 0, lc, 0)

def make_torus(R, r, index=0):
    gmsh.model.geo.addPoint(R, 0, 0, lc, 1 + 6 * index)
    gmsh.model.geo.addPoint(-R, 0, 0, lc, 2 + 6 * index)

    gmsh.model.geo.addPoint(R + r, 0, 0, lc, 3 + 6 * index)
    gmsh.model.geo.addPoint(R - r, 0, 0, lc, 4 + 6 * index)

    gmsh.model.geo.addPoint(-R - r, 0, 0, lc, 5 + 6 * index)
    gmsh.model.geo.addPoint(-R + r, 0, 0, lc, 6 + 6 * index)

    gmsh.model.geo.addCircleArc(3 + 6 * index, 1 + 6 * index, 4 + 6 * index, 1 + 8 * index, 0, 1, 0)
    gmsh.model.geo.addCircleArc(4 + 6 * index, 1 + 6 * index, 3 + 6 * index, 2 + 8 * index, 0, 1, 0)

    gmsh.model.geo.addCircleArc(5 + 6 * index, 2 + 6 * index, 6 + 6 * index, 3 + 8 * index, 0, 1, 0)
    gmsh.model.geo.addCircleArc(6 + 6 * index, 2 + 6 * index, 5 + 6 * index, 4 + 8 * index, 0, 1, 0)

    gmsh.model.geo.addCircleArc(3 + 6 * index, 0, 5 + 6 * index, 5 + 8 * index)
    gmsh.model.geo.addCircleArc(5 + 6 * index, 0, 3 + 6 * index, 6 + 8 * index)

    gmsh.model.geo.addCircleArc(4 + 6 * index, 0, 6 + 6 * index, 7 + 8 * index)
    gmsh.model.geo.addCircleArc(6 + 6 * index, 0, 4 + 6 * index, 8 + 8 * index)

    gmsh.model.geo.addCurveLoop([1 + 8 * index, -8 - 8 * index, 4 + 8 * index, 6 + 8 * index], 1 + 4 * index)
    gmsh.model.geo.addCurveLoop([1 + 8 * index, 7 + 8 * index, 4 + 8 * index, -5 - 8 * index], 2 + 4 * index)
    gmsh.model.geo.addCurveLoop([-2 - 8 * index, -8 - 8 * index, -3 - 8 * index, 6 + 8 * index], 3 + 4 * index)
    gmsh.model.geo.addCurveLoop([-2 - 8 * index, 7 + 8 * index, -3 - 8 * index, -5 - 8 * index], 4 + 4 * index)

    gmsh.model.geo.addSurfaceFilling([1 + 4 * index], 1 + 4 * index)
    gmsh.model.geo.addSurfaceFilling([2 + 4 * index], 2 + 4 * index)
    gmsh.model.geo.addSurfaceFilling([3 + 4 * index], 3 + 4 * index)
    gmsh.model.geo.addSurfaceFilling([4 + 4 * index], 4 + 4 * index)


make_torus(1, .4, 0)
make_torus(1, .2, 1)

l = gmsh.model.geo.addSurfaceLoop([i + 1 for i in range(8)])
gmsh.model.geo.addVolume([l])

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(3)

gmsh.write("torus.msh")
gmsh.write("torus.geo_unrolled")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()