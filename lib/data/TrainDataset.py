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