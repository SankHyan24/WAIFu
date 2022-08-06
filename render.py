from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import trimesh
import os

def load_trimesh(root_dir):
    folders = os.listdir(root_dir)
    meshs = trimesh.load(os.path.join(root_dir, 'model.obj'))

    return meshs

class train:
    def __init__(self) -> None:
        self.mesh_dic=load_trimesh("./Datas")
    
a=train()
a.mesh_dic.show()

# def Draw():
#     glClear(GL_COLOR_BUFFER_BIT)
#     glRotatef(0.5, 0, 1, 0)
#     glutWireTeapot(0.5)
#     glFlush()
# glutInit()
# glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
# glutInitWindowSize(400, 400)
# glutCreateWindow("test")
# glutDisplayFunc(Draw)
# glutIdleFunc(Draw)
# glutMainLoop() 