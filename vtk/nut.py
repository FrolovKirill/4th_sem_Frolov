import vtk
import numpy as np
import gmsh
import math
import os

wave_vec = np.ones(shape=(1, 3), dtype=np.double) * 2
omega = 0.5

# Класс расчётной сетки
class CalcMesh:

    # Конструктор сетки, полученной из stl-файла
    def __init__(self, nodes_coords, tetrs_points):
        self.time = 0
        self.nodes = np.array([nodes_coords[0::3], nodes_coords[1::3], nodes_coords[2::3]])

        self.smth = np.sin(- wave_vec @ self.nodes)

        self.velocity = np.zeros(shape=(3, int(len(nodes_coords) / 3)), dtype=np.double)
        self.velocity[2] = np.ones(int(len(nodes_coords) / 3), dtype=np.double) * (-1.5)
        self.velocity[0] = self.nodes[1] * 1
        self.velocity[1] = -self.nodes[0] * 1

        self.tetrs = np.array([tetrs_points[0::4], tetrs_points[1::4], tetrs_points[2::4], tetrs_points[3::4]])
        self.tetrs -= 1


    # Метод отвечает за выполнение для всей сетки шага по времени величиной tau и за изменение скорости
    def move(self, tau):
        self.nodes += self.velocity * tau
        self.time += tau
        self.velocity[0] = self.nodes[1] * 1
        self.velocity[1] = -self.nodes[0] * 1
        self.smth = np.sin(omega * self.time - wave_vec @ self.nodes)

    # Метод отвечает за запись текущего состояния сетки в снапшот в формате VTK
    def snapshot(self, snap_number):
        # Сетка в терминах VTK
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        # Точки сетки в терминах VTK
        points = vtk.vtkPoints()

        # Скалярное поле на точках сетки
        smth = vtk.vtkDoubleArray()
        smth.SetName("smth")

        # Векторное поле на точках сетки
        vel = vtk.vtkDoubleArray()
        vel.SetNumberOfComponents(3)
        vel.SetName("vel")

        # Обходим все точки нашей расчётной сетки
        for i in range(0, len(self.nodes[0])):
            # Вставляем новую точку в сетку VTK-снапшота
            points.InsertNextPoint(self.nodes[0, i], self.nodes[1, i], self.nodes[2, i])
            # Добавляем значение скалярного поля в этой точке
            smth.InsertNextValue(self.smth[0][i])
            # Добавляем значение векторного поля в этой точке
            vel.InsertNextTuple((self.velocity[0, i], self.velocity[1, i], self.velocity[2, i]))

        # Грузим точки в сетку
        unstructuredGrid.SetPoints(points)

        # Присоединяем векторное и скалярное поля к точкам
        unstructuredGrid.GetPointData().AddArray(smth)
        unstructuredGrid.GetPointData().AddArray(vel)

        # А теперь пишем, как наши точки объединены в тетраэдры
        for i in range(0, len(self.tetrs[0])):
            tetr = vtk.vtkTetra()
            for j in range(0, 4):
                tetr.GetPointIds().SetId(j, self.tetrs[j, i])
            unstructuredGrid.InsertNextCell(tetr.GetCellType(), tetr.GetPointIds())

        # Создаём снапшот в файле с заданным именем
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputDataObject(unstructuredGrid)
        writer.SetFileName("nut_1-" + str(snap_number) + ".vtu")
        writer.Write()


# Построением сетки средствами gmsh
gmsh.initialize()

# Считаем STL
try:
    path = os.path.dirname(os.path.abspath(__file__))
    gmsh.merge(os.path.join(path, 'Nut.STL'))
except:
    print("Could not load STL mesh: bye!")
    gmsh.finalize()
    exit(-1)

# Восстановим геометрию
angle = 5
forceParametrizablePatches = True
includeBoundary = False
curveAngle = 180
gmsh.model.mesh.removeDuplicateNodes()
gmsh.model.mesh.classifySurfaces(angle * math.pi / 180., includeBoundary, forceParametrizablePatches, curveAngle * math.pi / 180.)
gmsh.model.mesh.createGeometry()

# Зададим объём по считанной поверхности
s = gmsh.model.getEntities(2)
l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
gmsh.model.geo.addVolume([l])

gmsh.model.geo.synchronize()

# Зададим мелкость желаемой сетки
f = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(f, "F", "4")
gmsh.model.mesh.field.setAsBackgroundMesh(f)

# Построим сетку
gmsh.model.mesh.generate(3)

# Теперь извлечём из gmsh данные об узлах сетки
nodeTags, nodesCoord, parametricCoord = gmsh.model.mesh.getNodes()

# И данные об элементах сетки тоже извлечём, нам среди них нужны только тетраэдры, которыми залит объём
GMSH_TETR_CODE = 4
tetrsNodesTags = None
elementTypes, elementTags, elementNodeTags = gmsh.model.mesh.getElements()
for i in range(0, len(elementTypes)):
    if elementTypes[i] != GMSH_TETR_CODE:
        continue
    tetrsNodesTags = elementNodeTags[i]

if tetrsNodesTags is None:
    print("Can not find tetra data. Exiting.")
    gmsh.finalize()
    exit(-2)

print("The model has %d nodes and %d tetrs" % (len(nodeTags), len(tetrsNodesTags) / 4))

# На всякий случай проверим, что номера узлов идут подряд и без пробелов
for i in range(0, len(nodeTags)):
    assert (i == nodeTags[i] - 1)
assert(len(tetrsNodesTags) % 4 == 0)

mesh = CalcMesh(nodesCoord, tetrsNodesTags)
mesh.nodes[2] += 10
mesh.nodes[0] -= np.array(mesh.nodes[0]).mean()
mesh.nodes[1] -= np.array(mesh.nodes[1]).mean()

mesh.snapshot(0)
k = 4
for i in range(1, 100 * k):
    mesh.move(0.1 / k)
    mesh.snapshot(i)
    print('snapshot ' + str(i) + ';')

gmsh.finalize()
