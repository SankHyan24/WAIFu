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
    def __len__(self)->int:
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)
    def show_all_meshes(self):
        for i in self.mesh_dic:
            i.show()
    def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        pitch=self.pitch_list[pid]
    def get_item(self, index):
        subject = self.subjects[index % len(self.subjects)]
        yaw = self.yaw_list[index % len(self.yaw_list)]
        pitch = self.pitch_list[index % len(self.pitch_list)]
        return subject, yaw, pitch
    def __getitem__(self, index):
        return self.get_item(index)





a=TrainDataset()
a.show_all_meshes()
