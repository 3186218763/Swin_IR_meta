import torch
import torch.nn.functional as F
import torch.nn as nn


def laplacian_pyramid(x):
    # x: [B, C, H, W]

    gaussian = F.avg_pool2d(x, kernel_size=2, stride=2)
    up = F.interpolate(gaussian, size=x.shape[-2:], mode='bilinear', align_corners=False)

    lap = x - up
    return lap

# 拉普拉斯
def lpss_loss(pred, target):
    pred_lap = laplacian_pyramid(pred)
    target_lap = laplacian_pyramid(target)

    return F.l1_loss(pred_lap, target_lap)

# 频域
def feq_loss(pred, target):
    pred_fft = torch.fft.rfft2(pred, norm='ortho')
    target_fft = torch.fft.rfft2(target, norm='ortho')

    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    return F.l1_loss(pred_mag, target_mag)

# 物理条件
def order_constraint_loss(pred):
    """
    pred: [B, 3, H, W]
          0=tmax, 1=tmean, 2=tmin
    """

    tmax = pred[:, 0]
    tmean = pred[:, 1]
    tmin = pred[:, 2]

    loss1 = F.relu(tmean - tmax)
    loss2 = F.relu(tmin - tmean)

    return loss1.mean() + loss2.mean()


# ssim
def ssim_loss(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)

    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    return 1 - ssim.mean()

# 锋面
def gradient_loss(pred, target):
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]

    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]

    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)



class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, pred, target):

        # ===== 基础 loss =====
        l1 = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        ssim = ssim_loss(pred, target)
        grad = gradient_loss(pred, target)
        physical = order_constraint_loss(pred)



        total_loss = (
                1.0 * l1 +
                1.0 * mse +
                0.6 * ssim +
                0.2 * grad +
                0.1 * physical
        )



        return total_loss


if __name__ == '__main__':

    """
| Loss     | 作用          
| -------- | ----------- 
| L1       | 数值拟合（主干）    
| MSE      | 平滑约束        
| SSIM     | 空间结构  
| Gradient | 锋面/梯度
| Physical | 物理约束 

    """
    pred = torch.randn(16, 3, 80, 130)
    target = torch.randn(16, 3, 80, 130)
    loss_fn = MultiTaskLoss()
    loss = loss_fn(pred, target)
    print(loss)