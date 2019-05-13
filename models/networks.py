# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import functools

###############################################################################
# Functions
###############################################################################
EPSILON = 1e-6


def gradient(input, do_normalize=False):
    if input.dim() == 2:
        D_ry = input[1:, :]
        D_ly = input[:-1, :]
        D_rx = input[:, 1:]
        D_lx = input[:, :-1]
    elif input.dim() == 3:
        D_ry = input[:, 1:, :]
        D_ly = input[:, :-1, :]
        D_rx = input[:, :, 1:]
        D_lx = input[:, :, :-1]
    elif input.dim() == 4:
        D_ry = input[:, :, 1:, :]
        D_ly = input[:, :, :-1, :]
        D_rx = input[:, :, :, 1:]
        D_lx = input[:, :, :, :-1]

    Dx = D_rx - D_lx
    Dy = D_ry - D_ly
    if do_normalize:
        Dx = Dx / (D_rx + D_lx + EPSILON)
        Dy = Dy / (D_ry + D_ly + EPSILON)
    return Dx, Dy


def init_weights(net, init_type='normal', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or
                                     classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count -
                             opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_epoch, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented',
                                   opt.lr_policy)
    return scheduler


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def print_network(net_):
    num_params = 0
    for param in net_.parameters():
        num_params += param.numel()
    print(net_)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


class LaplacianLayer(nn.Module):

    def __init__(self):
        super(LaplacianLayer, self).__init__()
        w_nom = torch.FloatTensor([[0, -1, 0], [-1, 4, -1],
                                   [0, -1, 0]]).view(1, 1, 3, 3).cuda()
        w_den = torch.FloatTensor([[0, 1, 0], [1, 4, 1],
                                   [0, 1, 0]]).view(1, 1, 3, 3).cuda()
        self.register_buffer('w_nom', w_nom)
        self.register_buffer('w_den', w_den)

    def forward(self, input, do_normalize=True):
        assert (input.dim() == 2 or input.dim() == 3 or input.dim() == 4)
        input_size = input.size()
        if input.dim() == 4:
            x = input.view(input_size[0] * input_size[1], 1, input_size[2],
                           input_size[3])
        elif input.dim() == 3:
            x = input.unsqueeze(1)
        else:
            x = input.unsqueeze(0).unsqueeze(0)
        x_nom = torch.nn.functional.conv2d(
            input=x, weight=autograd.Variable(self.w_nom), stride=1, padding=0)
        if do_normalize:
            x_den = torch.nn.functional.conv2d(
                input=x, weight=autograd.Variable(self.w_den), stride=1, padding=0)
            # x_den = x.std() + 1e-5
            x = (x_nom.abs() / x_den)
        else:
            x = x_nom.abs()
        if input.dim() == 4:
            return x.view(input_size[0], input_size[1], input_size[2] - 2,
                          input_size[3] - 2)
        elif input.dim() == 3:
            return x.squeeze(1)
        elif input.dim() == 2:
            return x.squeeze(0).squeeze(0)


class JointLoss(nn.Module):

    def __init__(self, opt):
        super(JointLoss, self).__init__()
        self.opt = opt
        self.w_si_mse = 1.0
        self.w_l1_rel = 1.0
        self.w_confidence = 1.0
        self.w_grad = 0.75
        self.w_sm = 0.1
        self.w_sm1 = 0.075
        self.w_sm2 = 0.1
        self.w_normal = 0.5
        self.num_scales = 5
        self.total_loss = None
        self.laplacian_func = LaplacianLayer()

    def LaplacianSmoothnessLoss(self, depth, img):
        img_lap = self.laplacian_func(img, do_normalize=False)
        depth_lap = self.laplacian_func(depth, do_normalize=False)

        x = (-img_lap.mean(1)).exp() * (depth_lap)
        return x.mean()

    def compute_image_aware_2nd_smoothness_cost(self, depth, img):
        depth_grad_x, depth_grad_y = gradient(depth, do_normalize=False)
        depth_grad_x2, depth_grad_xy = gradient(
            depth_grad_x, do_normalize=False)
        depth_grad_yx, depth_grad_y2 = gradient(
            depth_grad_y, do_normalize=False)
        return depth_grad_x2.abs().mean() \
            + depth_grad_xy.abs().mean() + depth_grad_yx.abs().mean() + \
            depth_grad_y2.abs().mean()

    def compute_image_aware_1st_smoothness_cost(self, depth, img):
        depth_grad_x, depth_grad_y = gradient(depth, do_normalize=False)
        img_grad_x, img_grad_y = gradient(img, do_normalize=False)
        if img.dim() == 3:
            weight_x = torch.exp(-img_grad_x.abs().mean(0))
            weight_y = torch.exp(-img_grad_y.abs().mean(0))
            cost = ((depth_grad_x.abs() * weight_x)[:-1, :] +
                    (depth_grad_y.abs() * weight_y)[:, :-1]).mean()
        else:
            weight_x = torch.exp(-img_grad_x.abs().mean(1))
            weight_y = torch.exp(-img_grad_y.abs().mean(1))
            cost = ((depth_grad_x.abs() * weight_x)[:, :-1, :] +
                    (depth_grad_y.abs() * weight_y)[:, :, :-1]).mean()
        return cost

    def SecondOrderLoss(self, log_pred_d, mask, log_gt):
        N = torch.sum(mask) + 1e-6

        v_pred_lap = log_pred_d[:-2, :] - 2 * \
            log_pred_d[1:-1, :] + log_pred_d[2:, :]
        v_gt_lap = log_gt[:-2, :] - 2 * log_gt[1:-1, :] + log_gt[2:, :]

        v_diff = torch.abs(v_pred_lap - v_gt_lap)
        v_mask = torch.mul(torch.mul(mask[:-2, :], mask[2:, :]), mask[1:-1, :])
        v_lap_term = torch.mul(v_diff, v_mask)

        h_pred_lap = log_pred_d[:, :-2] - 2 * \
            log_pred_d[:, 1:-1] + log_pred_d[:, 2:]
        h_gt_lap = log_gt[:, :-2] - 2 * log_gt[:, 1:-1] + log_gt[:, 2:]

        h_diff = torch.abs(h_pred_lap - h_gt_lap)
        h_mask = torch.mul(torch.mul(mask[:, :-2], mask[:, 2:]), mask[:, 1:-1])
        h_lap_term = torch.mul(h_diff, h_mask)

        second_order_term = torch.sum(v_lap_term) + torch.sum(h_lap_term)
        second_order_term = second_order_term / N

        return second_order_term

    def GradientLoss(self, log_prediction_d, mask, log_gt):
        log_d_diff = log_prediction_d - log_gt

        v_gradient = torch.abs(log_d_diff[:, :-2, :] - log_d_diff[:, 2:, :])
        v_mask = torch.mul(mask[:, :-2, :], mask[:, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(log_d_diff[:, :, :-2] - log_d_diff[:, :, 2:])
        h_mask = torch.mul(mask[:, :, :-2], mask[:, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        N = torch.sum(h_mask) + torch.sum(v_mask) + EPSILON

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss / N

        return gradient_loss

    def DeMonGradientLoss(self, prediction_d, mask, gt_d):

        v_mask = torch.mul(mask[:, :-2, :], mask[:, 2:, :])
        v_grad_pred = (prediction_d[:, :-2, :] - prediction_d[:, 2:, :]) / (
            prediction_d[:, :-2, :] + prediction_d[:, 2:, :] + EPSILON)
        v_grad_gt = (gt_d[:, :-2, :] - gt_d[:, 2:, :]) / (
            gt_d[:, :-2, :] + gt_d[:, 2:, :] + EPSILON)
        v_grad_term = v_mask * torch.abs(v_grad_pred - v_grad_gt)

        h_mask = torch.mul(mask[:, :, :-2], mask[:, :, 2:])
        h_grad_pred = (prediction_d[:, :, :-2] - prediction_d[:, :, 2:]) / (
            prediction_d[:, :, :-2] + prediction_d[:, :, 2:] + EPSILON)
        h_grad_gt = (gt_d[:, :, :-2] - gt_d[:, :, 2:]) / (
            gt_d[:, :, :-2] + gt_d[:, :, 2:] + EPSILON)

        h_grad_term = h_mask * torch.abs(h_grad_pred - h_grad_gt)

        N = torch.sum(h_mask) + torch.sum(v_mask) + EPSILON

        gradient_loss = torch.sum(v_grad_term) + torch.sum(h_grad_term)
        gradient_loss = gradient_loss / N

        return gradient_loss

    def Data_Loss(self, log_prediction_d, mask, log_gt):
        N = torch.sum(mask) + EPSILON
        log_d_diff = log_prediction_d - log_gt
        log_d_diff = torch.mul(log_d_diff, mask)
        s1 = torch.sum(torch.pow(log_d_diff, 2)) / N
        s2 = (torch.sum(log_d_diff) * torch.sum(log_d_diff)) / (N * N)

        data_loss = s1 - s2

        return data_loss

    def Confidence_Loss(self, pred_confidence, mask, pred_d, gt_d):
        # using least square to find scaling factor
        N = torch.sum(mask) + EPSILON
        N = N.item()

        if N > 0.5:
            scale_factor = torch.median(
                gt_d.data[mask.data > 0.1] /
                (pred_d.data[mask.data > 0.1] + EPSILON)).item()
            pred_d_aligned = pred_d * scale_factor

            error = torch.abs(pred_d_aligned.data -
                              gt_d.data) / (gt_d.data + EPSILON)
            error = torch.exp(-error * 2.0)

            error_var = autograd.Variable(error, requires_grad=False)
            u_loss = mask * torch.abs(pred_confidence - error_var)
            confidence_term = torch.sum(u_loss) / N
        else:
            confidence_term = 0.0

        return confidence_term

    def Normal_Loss(self, pred_d, gt_d, mask, normalized_p_3d, p_3d_gt):

        p_3d_pred = normalized_p_3d * pred_d.unsqueeze(1).repeat(1, 2, 1, 1)

        x_mask = mask[:, 1:-1, :-2] * mask[:, 1:-1, 2:]
        y_mask = mask[:, :-2, 1:-1] * mask[:, 2:, 1:-1]

        final_mask = y_mask * x_mask

        grad_x_pred = (pred_d[:, 1:-1, :-2] - pred_d[:, 1:-1, 2:]) / (
            p_3d_pred[:, 0, 1:-1, :-2] - p_3d_pred[:, 0, 1:-1, 2:] + EPSILON)
        grad_x_gt = (gt_d[:, 1:-1, :-2] - gt_d[:, 1:-1, 2:]) / (
            p_3d_gt[:, 0, 1:-1, :-2] - p_3d_gt[:, 0, 1:-1, 2:] + EPSILON)

        grad_y_pred = (pred_d[:, :-2, 1:-1] - pred_d[:, 2:, 1:-1]) / (
            p_3d_pred[:, 1, :-2, 1:-1] - p_3d_pred[:, 1, 2:, 1:-1] + EPSILON)
        grad_y_gt = (gt_d[:, :-2, 1:-1] - gt_d[:, 2:, 1:-1]) / (
            p_3d_gt[:, 1, :-2, 1:-1] - p_3d_gt[:, 1, 2:, 1:-1] + EPSILON)

        norm_pred = torch.sqrt(grad_x_pred * grad_x_pred +
                               grad_y_pred * grad_y_pred + 1.0) + EPSILON
        norm_gt = torch.sqrt(grad_x_gt * grad_x_gt + grad_y_gt * grad_y_gt +
                             1.0) + EPSILON

        dot_product = grad_x_gt * grad_x_pred + grad_y_gt * grad_y_pred + 1.0
        dot_product = dot_product * final_mask

        N = torch.sum(final_mask) + EPSILON

        normal_term = dot_product / (norm_pred * norm_gt)

        normal_term = 1.0 - torch.sum(normal_term) / N

        return normal_term

    def Weighted_Data_Loss(self, log_prediction_d, w_mask, log_gt):
        log_d_diff = log_prediction_d - log_gt

        wx_2 = torch.sum(w_mask * torch.pow(log_d_diff, 2))
        wx = torch.sum(w_mask * log_d_diff)
        w_sum = torch.sum(w_mask)
        total_term = w_sum * wx_2 - torch.pow(wx, 2)
        N = w_sum * w_sum + EPSILON

        return total_term / N

    def L1_inv_loss(self, pred_d_aligned, mask, depth_gt):

        l1_inv_error = torch.abs(1. / (depth_gt + EPSILON) - 1. /
                                 (EPSILON + pred_d_aligned))
        l1_inv_error = l1_inv_error * mask
        num_valid_pixels = torch.sum(mask) + EPSILON

        return torch.sum(l1_inv_error) / num_valid_pixels

    def L1RelLoss(self, pred_d_aligned, mask, depth_gt):
        l1_rel_error = torch.abs(depth_gt - pred_d_aligned)
        l1_rel_error = l1_rel_error * mask
        l1_rel_error = l1_rel_error / \
            torch.abs(depth_gt + pred_d_aligned + 1e-8)

        num_valid_pixels = torch.sum(mask) + EPSILON
        return torch.sum(l1_rel_error) / num_valid_pixels

    def L1_rel_loss(self, pred_d_aligned, mask, depth_gt):

        l1_rel_error = torch.abs(depth_gt - pred_d_aligned)
        l1_rel_error = l1_rel_error * mask
        l1_rel_error = l1_rel_error / torch.abs(depth_gt + 1e-8)

        num_valid_pixels = torch.sum(mask) + EPSILON

        return torch.sum(l1_rel_error) / num_valid_pixels

    def compute_si_rmse(self, pred_log_d, targets):
        gt_mask = targets['gt_mask'].cuda()
        log_d_gt = torch.log(targets['depth_gt'].cuda())
        env_mask = targets['env_mask'].cuda()

        human_gt_mask = (1.0 - env_mask) * gt_mask
        env_gt_mask = env_mask * gt_mask

        si_rmse_full = 0.0
        si_rmse_human = 0.0
        si_rmse_env = 0.0

        si_rmse_inter = 0.0
        si_rmse_intra = 0.0

        # compute full error
        for i in range(0, gt_mask.size(0)):

            mse_full = self.Data_Loss(pred_log_d[i, :, :], gt_mask[i, :, :],
                                      log_d_gt[i, :, :])
            mse_env = self.Data_Loss(pred_log_d[i, :, :], env_gt_mask[i, :, :],
                                     log_d_gt[i, :, :])
            mse_intra = self.Data_Loss(pred_log_d[i, :, :], human_gt_mask[i, :, :],
                                       log_d_gt[i, :, :])

            # compute human error
            n_full = torch.sum(gt_mask[i, :, :])
            n_human = torch.sum(human_gt_mask[i, :, :])
            n_env = torch.sum(env_gt_mask[i, :, :])

            log_diff = pred_log_d[i, :, :] - log_d_gt[i, :, :]
            log_diff_mask = log_diff * gt_mask[i, :, :]

            # full human error
            sum_sq_log_diff = torch.sum(torch.pow(log_diff_mask, 2))
            sum_log_diff = torch.sum(log_diff_mask)

            per_pixel_error = n_full * torch.pow(
                log_diff, 2) + sum_sq_log_diff - 2 * sum_log_diff * log_diff
            per_pixel_error = per_pixel_error * human_gt_mask[i, :, :]

            mse_human = torch.sum(per_pixel_error) / \
                (n_human * n_full + EPSILON)

            # inter class mse error
            log_diff_env_mask = log_diff * env_gt_mask[i, :, :]
            sum_sq_log_env_diff = torch.sum(torch.pow(log_diff_env_mask, 2))
            sum_log_env_diff = torch.sum(log_diff_env_mask)

            inter_error = n_env * torch.pow(
                log_diff, 2) + sum_sq_log_env_diff - 2 * sum_log_env_diff * log_diff
            inter_error = inter_error * human_gt_mask[i, :, :]

            mse_inter = torch.sum(inter_error) / (n_human * n_env + EPSILON)

            si_rmse_full += torch.sqrt(2.0 * mse_full)
            si_rmse_human += torch.sqrt(mse_human)
            si_rmse_env += torch.sqrt(2.0 * mse_env)
            si_rmse_intra += torch.sqrt(2.0 * mse_intra)
            si_rmse_inter += torch.sqrt(mse_inter)

        return si_rmse_full, si_rmse_human, si_rmse_env, si_rmse_intra, si_rmse_inter

    def compute_l1_rel_error(self, pred_d, targets):
        gt_mask = targets['gt_mask']
        d_gt = targets['depth_gt']

        rel_full = 0.

        for i in range(0, gt_mask.size(0)):
            gt_d_np = d_gt[i, :, :].cpu().numpy()
            pred_d_np = pred_d[i, :, :].cpu().numpy()
            gt_mask_np = gt_mask[i, :, :].cpu().numpy()

            scale_factor = np.linalg.lstsq(
                np.expand_dims(pred_d_np[gt_mask_np > 1e-8], axis=-1),
                gt_d_np[gt_mask_np > 1e-8])
            scale_factor = scale_factor[0][0]

            pred_d_aligned_np = pred_d_np * scale_factor

            total_full_rel = np.sum(gt_mask_np * np.abs(gt_d_np - pred_d_aligned_np) /
                                    (gt_d_np + EPSILON))
            rel_full += total_full_rel / (np.sum(gt_mask_np) + EPSILON)

        return rel_full

    def compute_rmse_error(self, pred_d, targets):
        gt_mask = targets['gt_mask']
        d_gt = targets['depth_gt']

        rmse_full = 0.

        for i in range(0, gt_mask.size(0)):
            gt_d_np = d_gt[i, :, :].cpu().numpy()
            pred_d_np = pred_d[i, :, :].cpu().numpy()
            gt_mask_np = gt_mask[i, :, :].cpu().numpy()

            scale_factor = np.linalg.lstsq(
                np.expand_dims(pred_d_np[gt_mask_np > 1e-8], axis=-1),
                gt_d_np[gt_mask_np > 1e-8])
            scale_factor = scale_factor[0][0]

            pred_d_aligned_np = pred_d_np * scale_factor

            total_full_rmse = np.sum(gt_mask_np *
                                     np.square(gt_d_np - pred_d_aligned_np))
            rmse_full += np.sqrt(total_full_rmse /
                                 (np.sum(gt_mask_np) + EPSILON))

        return rmse_full

    def Data_Human_Loss(self, pred_log_d, gt_mask, human_gt_mask, log_d_gt):
        n_full = torch.sum(gt_mask)
        n_human = torch.sum(human_gt_mask)

        log_diff = pred_log_d - log_d_gt
        log_diff_mask = log_diff * gt_mask

        sum_sq_log_diff = torch.sum(torch.pow(log_diff_mask, 2))
        sum_log_diff = torch.sum(log_diff_mask)

        inter_error = n_full * torch.pow(
            log_diff, 2) + sum_sq_log_diff - 2 * sum_log_diff * log_diff
        inter_error = inter_error * human_gt_mask

        mse_human = torch.sum(inter_error) / (n_human * n_full + EPSILON)
        mse_human = mse_human / 2.0
        return mse_human

    def __call__(self, input_images, log_pred_d_0, pred_confidence, targets):

        log_pred_d_1 = log_pred_d_0[:, ::2, ::2]
        log_pred_d_2 = log_pred_d_1[:, ::2, ::2]
        log_pred_d_3 = log_pred_d_2[:, ::2, ::2]
        log_pred_d_4 = log_pred_d_3[:, ::2, ::2]

        input_0 = input_images
        input_1 = input_0[:, :, ::2, ::2]
        input_2 = input_1[:, :, ::2, ::2]
        input_3 = input_2[:, :, ::2, ::2]
        input_4 = input_3[:, :, ::2, ::2]

        d_gt_0 = autograd.Variable(targets['depth_gt'].cuda(), requires_grad=False)
        log_d_gt_0 = torch.log(d_gt_0)
        log_d_gt_1 = log_d_gt_0[:, ::2, ::2]
        log_d_gt_2 = log_d_gt_1[:, ::2, ::2]
        log_d_gt_3 = log_d_gt_2[:, ::2, ::2]
        log_d_gt_4 = log_d_gt_3[:, ::2, ::2]

        gt_mask = autograd.Variable(targets['gt_mask'].cuda(), requires_grad=False)
        human_mask = 1.0 - \
            autograd.Variable(targets['env_mask'].cuda(), requires_grad=False)
        human_gt_mask = human_mask * gt_mask

        mask_0 = gt_mask
        mask_1 = mask_0[:, ::2, ::2]
        mask_2 = mask_1[:, ::2, ::2]
        mask_3 = mask_2[:, ::2, ::2]
        mask_4 = mask_3[:, ::2, ::2]

        data_term = 0.0
        grad_term = 0.0
        sm_term = 0.0
        confidence_term = 0.0

        num_samples = mask_0.size(0)

        for i in range(0, num_samples):
            if self.opt.human_data_term > 0.1:
                data_term += (self.w_si_mse / num_samples * self.Data_Loss(
                    log_pred_d_0[i, :, :], mask_0[i, :, :], log_d_gt_0[i, :, :]))
                data_term += (self.w_si_mse / num_samples * 0.5 * self.Data_Human_Loss(
                    log_pred_d_0[i, :, :], mask_0[i,
                                                  :, :], human_gt_mask[i, :, :],
                    log_d_gt_0[i, :, :]))
            else:
                data_term += (self.w_si_mse / num_samples * 1.5 * self.Data_Loss(
                    log_pred_d_0[i, :, :], mask_0[i, :, :], log_d_gt_0[i, :, :]))

        grad_term += self.w_grad * self.GradientLoss(log_pred_d_0, mask_0,
                                                     log_d_gt_0)
        grad_term += self.w_grad * self.GradientLoss(log_pred_d_1, mask_1,
                                                     log_d_gt_1)
        grad_term += self.w_grad * self.GradientLoss(log_pred_d_2, mask_2,
                                                     log_d_gt_2)
        grad_term += self.w_grad * self.GradientLoss(log_pred_d_3, mask_3,
                                                     log_d_gt_3)
        grad_term += self.w_grad * self.GradientLoss(log_pred_d_4, mask_4,
                                                     log_d_gt_4)

        sm_term += self.w_sm1 * self.compute_image_aware_1st_smoothness_cost(
            log_pred_d_0, input_0)
        sm_term += (self.w_sm1 * 0.5 * self.compute_image_aware_1st_smoothness_cost(
            log_pred_d_1, input_1))
        sm_term += (self.w_sm1 * 0.25 * self.compute_image_aware_1st_smoothness_cost(
            log_pred_d_2, input_2))
        sm_term += (self.w_sm1 * 0.125 * self.compute_image_aware_1st_smoothness_cost(
            log_pred_d_3, input_3))
        sm_term += (self.w_sm1 * 0.0625 * self.compute_image_aware_1st_smoothness_cost(
            log_pred_d_4, input_4))

        sm_term += self.w_sm2 * \
            self.LaplacianSmoothnessLoss(log_pred_d_0, input_0)
        sm_term += self.w_sm2 * 0.5 * self.LaplacianSmoothnessLoss(
            log_pred_d_1, input_1)
        sm_term += self.w_sm2 * 0.25 * self.LaplacianSmoothnessLoss(
            log_pred_d_2, input_2)
        sm_term += self.w_sm2 * 0.125 * self.LaplacianSmoothnessLoss(
            log_pred_d_3, input_3)
        sm_term += self.w_sm2 * 0.0625 * self.LaplacianSmoothnessLoss(
            log_pred_d_4, input_4)

        print('data_term %f' % data_term.item())
        print('grad_term %f' % grad_term.item())
        print('sm_term %f' % sm_term.item())

        total_loss = data_term + grad_term + sm_term + confidence_term

        self.total_loss = total_loss

        return total_loss.item()

    def get_loss_var(self):
        return self.total_loss
