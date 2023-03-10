import gmsh
import math
import os
import sys

gmsh.initialize()

path = os.path.dirname(os.path.abspath(__file__))
gmsh.merge(os.path.join(path, 'Ruchka_rotametra v1.stl'))

angle = 10

forceParametrizablePatches = True

includeBoundary = False

curveAngle = 180

gmsh.model.mesh.classifySurfaces(angle * math.pi / 180., includeBoundary,
                                 forceParametrizablePatches,
                                 curveAngle * math.pi / 180.)

gmsh.model.mesh.createGeometry()

s = gmsh.model.getEntities(2)
l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
gmsh.model.geo.addVolume([l])

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(3)
gmsh.write('Ruchka_rotametra v1.msh')

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
