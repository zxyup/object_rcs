import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from gwcnet import GwcNet_G
import torchvision.transforms as tf
# 归一化过程使用的均值方差，通过 ImageNet 的数据集计算得到
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# 当卷积尺寸不变的时候会自动寻找最优卷积算法
cudnn.benchmark = True

class DepthModel(object):
    
    def __init__(self, path: os.PathLike, maxdisp: int = 192):
        super().__init__()
        self.model = GwcNet_G(maxdisp).cuda()
        print(f'loading model from {path}')
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.tt = tf.ToTensor()
        self.norm = tf.Normalize(mean=mean, std=std)
        
    def forward(self, l, r):
        """l, r 对应左右图像，返回左图像的深度信息"""
        with torch.no_grad():
            l, r = self.tt(l).cuda(), self.tt(r).cuda()
            l, r = self.norm(l), self.norm(r)
            l, r = l.unsqueeze(dim=0), r.unsqueeze(dim=0)
            print(l.shape, r.shape)
            pred = self.model(l, r).squeeze(dim=0)
        return pred

    def __call__(self, l, r):
        """获取指定锚框中物体的深度"""
        return self.forward(l, r)

if __name__ == '__main__':
    model = DepthModel('./Gwc-Net-light.pth', maxdisp=192)
    # cs = PhotoStream("./left", "./right", "./maps")
    # for i, sample in enumerate(cs):
    #     start_time = time.time()
    #     disp_est_np = model.forward(sample['left'], sample['right'])
    #     print('Iter {}, time = {:3f}ms'.format(i, 1e3*(time.time() - start_time)))
    #     print('-'*20)
    #     disp_est_np = tensor2numpy(disp_est_np)
    #     # print(disp_est_np.dtype, disp_est_np.shape)
    #     # [top_pad:, :-right_pad]
    #     # cv.imshow('disp', disp_est_np/192)
    #     gray = (disp_est_np/192)
    #     cv.imshow("p", gray)
    #     if cv.waitKey(0) == ord('q'):
    #         break
    # cv.destroyAllWindows()