from torch import nn

from .DepthNormalizer import DepthNormalizer

from .HGFilters import *
from .SurfaceClassifier import *
import BaseWAIFuNet
class HGWAIFuNet(BaseWAIFuNet):
    def __init__(self, 
                 opt,
                 projection_mode='orthogonal'
                 ):
        super(BaseWAIFuNet, self).__init__(projection_mode=projection_mode)
    
        self.opt = opt
        # image encoder g, a FCN implemented with hourglass stack, 用于提取图片信息
        self.image_filter = HGFilter(opt)

        # f(F(x), z(X)) = s
        self.surface_classifier = SurfaceClassifier()

        # 用于转换z
        self.normalizer = DepthNormalizer(self.opt)

        self.im_feat_list = []
        # 下面两个看不懂不明白为什么网络内部通过卷积就可以得到
        # 暂时没有理会它们
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []

        init_net(self)
    
    def filter(self, images):
        '''
        从图像中提取信息
        :return: 图像信息概率热力图, tmpx, normx
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)

        # 非训练状态只取最后一个，即概率最大的
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]
    
    def query(self, points, calibs):

        xyz = self.projection(points, calibs)
        xy = xyz[:, :2, :] # 相机下的坐标
        z = xyz[:, 2:3, :]
        
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0) # 表征投影后是否在画面中

        z_feat = self.normalizer(z)

        for im_feat in self.im_feat_list:
            point_local_feat_list = [self.index(im_feat, xy), z_feat]

            point_local_feat = torch.cat(point_local_feat_list, dim=1)

            pred = in_img[:,None].float() * self.surface_classifier(point_local_feat)
            self.intermediate_preds_list.append(pred)
        
        self.preds = self.intermediate_preds_list[-1]
    
    def get_preds(self):
        return self.preds

    def forward(self, images, points, calibs):
        # 从图像中提取信息
        self.filter(images)

        self.query(points, calibs)

        return self.get_preds()

