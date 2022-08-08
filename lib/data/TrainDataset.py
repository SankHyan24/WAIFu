import os
import trimesh
import logging
from torch.utils.data import Dataset

log = logging.getLogger('trimesh')
log.setLevel(40)


def load_trimesh(root_dir):
    # load all the meshes in the root_dir
    file=os.listdir(root_dir)
    meshes=[]
    for i in file:
        meshes.append(trimesh.load(os.path.join(root_dir,i)))
    return meshes


class TrainDataset(Dataset):
    def __init__(self)->None:
        self.mesh_dic=load_trimesh("./Datas") # mesh list
        self.yaw_list = list(range(0,360,1))
        self.pitch_list = [0]
        self.subjects = self.get_subjects()

    def get_subjects(self):
        subjects = []
        for i in self.mesh_dic:
            subjects.append(i.metadata['subject'])
        return subjects
    def __len__(self):
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)
    def __len__(self)->int:
        return len(self.mesh_dic)
    def show_all_meshes(self):
        for i in self.mesh_dic:
            i.show() 





a=TrainDataset()
a.show_all_meshes()
