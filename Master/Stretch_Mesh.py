'''
Generation of the 'stretched mesh' used in solving the double-glazing problem using the finite-elements method. 
'''

from fenics import *
import numpy as np

def stretch_mesh(
        BottomPoint = (-1,-1),
        TopPoint = (1,1),
        nx = 32,
        ny = 32,
        s = 2.5
    ):
    '''
    Inputs:
        BottomPoint:    Tuple       Bottom left point of rectangle domain.
        TopPoint:       Tuple       Top right point of rectangle domain.
        nx:             Int         Number of vertices on horizontal axis.
        ny:             Int         Number of vertices on vertical axis.
        s:              Float       Stretching coefficient.
    Outputs:
        mesh:           Mesh        Mesh stretched towards right vertical boundary.
    '''
    # Create uniform rectangle mesh.
    a, b = BottomPoint[0], TopPoint[0]
    mesh = RectangleMesh(Point(a, BottomPoint[1]), Point(b, TopPoint[1]), nx, ny)
    ### Stretch horizontally to right vertical boundary.
    x = mesh.coordinates()[:,0]  
    y = mesh.coordinates()[:,1]
    # Stretching function.
    def denser(x,y):
        return [b + (a-b)*((x-a)/(b-a))**s, y]
    x_bar, y_bar = denser(x, y)
    xy_bar_coor = np.array([x_bar, y_bar]).transpose()
    mesh.coordinates()[:] = xy_bar_coor
    return mesh


if __name__ == "__main__":

    from vedo.dolfin import plot

    def main():
        mesh = stretch_mesh()
        plot(mesh, title="stretched mesh", interactive=True)
        return mesh

    mesh = main()