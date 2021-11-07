import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from yolox.utils import postprocess
from .basetrack import BaseTrack, TrackState

from cython_bbox import bbox_overlaps as bbox_ious

from scipy.optimize import linear_sum_assignment
import cv2
import random
seed = 0
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Remove randomness (may be slower on Tesla GPUs)
# https://pytorch.org/docs/stable/notes/randomness.html
if seed == 0:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

mse = torch.nn.MSELoss()
smoothL1 = torch.nn.SmoothL1Loss()

def bbox_dis(bbox1, bbox2):
    center1 = (bbox1[:, :2] + bbox1[:, 2:]) / 2
    center2 = (bbox2[:, :2] + bbox2[:, 2:]) / 2
    center1 = np.repeat(center1.reshape(-1, 1, 2), len(bbox2), axis=1)
    center2 = np.repeat(center2.reshape(1, -1, 2), len(bbox1), axis=0)
    dis = np.sqrt(np.sum((center1 - center2) ** 2, axis=-1))
    return dis

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    shared_kalman_ = KalmanFilter()

    def __init__(self, tlwh, score, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.exist_len = 0

        self.curr_tlbr = self.tlwh_to_tlbr(self._tlwh)

        self.det_dict = {}

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_predict_(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman_.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, track_id=None):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        if track_id:
            self.track_id = track_id['track_id']
            track_id['track_id'] += 1
        else:
            self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def activate_(self, kalman_filter, frame_id, track_id=None):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        if track_id:
            self.track_id = track_id['track_id']
            track_id['track_id'] += 1
        else:
            self.track_id = self.next_id_()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.curr_tlbr = self.tlwh_to_tlbr(new_track.tlwh)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.exist_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def re_activate_(self, new_track, frame_id, new_id=False):
        self.curr_tlbr = self.tlwh_to_tlbr(new_track.tlwh)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.exist_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id_()

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.exist_len += 1

        self.curr_tlbr = self.tlwh_to_tlbr(new_track.tlwh)
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(
            self,
            args,
            num_classes,
            confthre,
            nmsthre,
            convert_to_coco_format,
            frame_rate=30,
            model=None,
            model2=None,
            decoder=None,
            tracked_stracks=[],
            lost_stracks=[],
            removed_stracks=[],
            frame_id=0,
            ad_last_info={}
    ):
        self.model = model
        self.model_2 = model2 if model2 is not None else copy.deepcopy(model)
        self.decoder = decoder
        self.num_classes = num_classes
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.convert_to_coco_format = convert_to_coco_format

        self.tracked_stracks = copy.deepcopy(tracked_stracks)  # type: list[STrack]
        self.lost_stracks = copy.deepcopy(lost_stracks)  # type: list[STrack]
        self.removed_stracks = copy.deepcopy(removed_stracks)  # type: list[STrack]

        self.tracked_stracks_ = copy.deepcopy(tracked_stracks)  # type: list[STrack]
        self.lost_stracks_ = copy.deepcopy(tracked_stracks)  # type: list[STrack]
        self.removed_stracks_ = copy.deepcopy(tracked_stracks)  # type: list[STrack]

        self.frame_id = frame_id
        self.frame_id_ = frame_id
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.FRAME_THR = 10
        self.ATTACK_IOU_THR = 0.2
        self.attack_iou_thr = self.ATTACK_IOU_THR
        self.ad_last_info = copy.deepcopy(ad_last_info)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda()

    def recoverNoise(self, noise, img0):
        height = 800
        width = 1440
        noise = noise * self.std.view(1, -1, 1, 1)
        shape = img0.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)

        noise = noise[:, :, top:height - bottom, left:width - right]
        h, w, _ = img0.shape
        noise = self.resizeTensor(noise, h, w).cpu().squeeze().permute(1, 2, 0).numpy()

        noise = (noise[:, :, ::-1] * 255).astype(np.int)

        return noise

    @staticmethod
    def resizeTensor(tensor, height, width):
        h = torch.linspace(-1, 1, height).view(-1, 1).repeat(1, width).to(tensor.device)
        w = torch.linspace(-1, 1, width).repeat(height, 1).to(tensor.device)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0)

        output = F.grid_sample(tensor, grid=grid, mode='bilinear', align_corners=True)
        return output

    def ifgsm_adam_sg(
            self,
            imgs,
            img_info,
            dets,
            dets_second,
            outputs_index_1,
            outputs_index_2,
            last_info,
            outputs_ori,
            attack_id,
            attack_ind,
            target_id,
            target_ind
    ):
        img0_h = img_info[0][0].item()
        img0_w = img_info[1][0].item()
        H, W = imgs.size()[2:]
        r_w, r_h = img0_w / W, img0_h / H
        r_max = max(r_w, r_h)
        noise = torch.zeros_like(imgs)
        imgs_ori = imgs.clone().data
        outputs = outputs_ori
        reg = outputs[:, :4].clone().data
        reg_wh = reg[:, 2:] - reg[:, :2]

        strack_pool = copy.deepcopy(last_info['last_strack_pool'])
        last_attack_det = None
        last_target_det = None
        STrack.multi_predict(strack_pool)
        for strack in strack_pool:
            if strack.track_id == attack_id:
                last_attack_det = torch.from_numpy(strack.tlbr).cuda().float()
                last_attack_det[[0, 2]] = (last_attack_det[[0, 2]] - 0.5 * W * (r_w - r_max)) / r_max
                last_attack_det[[1, 3]] = (last_attack_det[[1, 3]] - 0.5 * H * (r_h - r_max)) / r_max
            elif strack.track_id == target_id:
                last_target_det = torch.from_numpy(strack.tlbr).cuda().float()
                last_target_det[[0, 2]] = (last_target_det[[0, 2]] - 0.5 * W * (r_w - r_max)) / r_max
                last_target_det[[1, 3]] = (last_target_det[[1, 3]] - 0.5 * H * (r_h - r_max)) / r_max
        last_attack_det_center = torch.round(
            (last_attack_det[:2] + last_attack_det[2:]) / 2) if last_attack_det is not None else None
        last_target_det_center = torch.round(
            (last_target_det[:2] + last_target_det[2:]) / 2) if last_target_det is not None else None

        dets_all = np.concatenate([dets, dets_second])
        hm_index = torch.cat([outputs_index_1, outputs_index_2])
        hm_index_ori = copy.deepcopy(hm_index)

        Ws = [W//s for s in [8, 16, 32]]
        Hs = [H//s for s in [8, 16, 32]]

        i = 0
        j = -1
        suc = True
        ori_index = []
        att_index = []
        attack_det_center_max = None
        target_det_center_max = None
        attack_outputs_ind = hm_index[attack_ind].clone()
        target_outputs_ind = hm_index[target_ind].clone()

        attack_i = None
        target_i = None
        for o_i in range(3):
            if attack_outputs_ind >= Ws[o_i] * Hs[o_i]:
                attack_outputs_ind -= Ws[o_i] * Hs[o_i]
            else:
                attack_i = o_i
                break
        for o_i in range(3):
            if target_outputs_ind >= Ws[o_i] * Hs[o_i]:
                target_outputs_ind -= Ws[o_i] * Hs[o_i]
            else:
                target_i = o_i
                break

        assert attack_i is not None and target_i is not None
        ori_index = []
        att_index = []
        while True:
            i += 1

            if i in [1, 10, 20, 30, 40, 45, 50, 55]:
                att_index_new = []
                attack_det_center = torch.stack([attack_outputs_ind % Ws[attack_i], attack_outputs_ind // Ws[attack_i]]).float().cuda()
                target_det_center = torch.stack([target_outputs_ind % Ws[target_i], target_outputs_ind // Ws[target_i]]).float().cuda()
                if attack_det_center_max is None:
                    attack_det_center_max = torch.round(attack_det_center * 2 ** attack_i).int()
                    attack_det_center_mid = torch.round(attack_det_center_max / 2).int()
                    attack_det_center_min = torch.round(attack_det_center_mid / 2).int()
                    attack_outputs_ind_max_ori = (attack_det_center_max[0] + attack_det_center_max[1] * Ws[0]).clone()
                    attack_outputs_ind_mid_ori = (attack_det_center_mid[0] + attack_det_center_mid[1] * Ws[1]
                                                  + Ws[0] * Hs[0]).clone()
                    attack_outputs_ind_min_ori = (attack_det_center_min[0] + attack_det_center_min[1] * Ws[2]
                                                  + Ws[0] * Hs[0] + Ws[1] * Hs[1]).clone()
                    ori_index.extend([
                        attack_outputs_ind_max_ori,
                        attack_outputs_ind_mid_ori,
                        attack_outputs_ind_min_ori
                    ])
                if target_det_center_max is None:
                    target_det_center_max = torch.round(target_det_center * 2 ** target_i).int()
                    target_det_center_mid = torch.round(target_det_center_max / 2).int()
                    target_det_center_min = torch.round(target_det_center_mid / 2).int()
                    target_outputs_ind_max_ori = (target_det_center_max[0] + target_det_center_max[1] * Ws[0]).clone()
                    target_outputs_ind_mid_ori = (target_det_center_mid[0] + target_det_center_mid[1] * Ws[1]
                                                  + Ws[0] * Hs[0]).clone()
                    target_outputs_ind_min_ori = (target_det_center_min[0] + target_det_center_min[1] * Ws[2]
                                                  + Ws[0] * Hs[0] + Ws[1] * Hs[1]).clone()
                    ori_index.extend([
                        target_outputs_ind_max_ori,
                        target_outputs_ind_mid_ori,
                        target_outputs_ind_min_ori
                    ])
                if last_target_det_center is not None:
                    last_target_det_center_ = last_target_det_center * Ws[0]/W
                    attack_center_delta = attack_det_center_max - last_target_det_center_
                    if torch.max(torch.abs(attack_center_delta)) > 1:
                        attack_center_delta /= torch.max(torch.abs(attack_center_delta))
                        attack_det_center_max = torch.round(attack_det_center_max - attack_center_delta).int()
                        attack_det_center_mid = torch.round(attack_det_center_max / 2).int()
                        attack_det_center_min = torch.round(attack_det_center_mid / 2).int()
                        attack_outputs_ind_max = attack_det_center_max[0] + attack_det_center_max[1] * Ws[0]
                        attack_outputs_ind_mid = attack_det_center_mid[0] + attack_det_center_mid[1] * Ws[1] \
                                                 + Ws[0] * Hs[0]
                        attack_outputs_ind_min = attack_det_center_min[0] + attack_det_center_min[1] * Ws[2] \
                                                 + Ws[0] * Hs[0] + Ws[1] * Hs[1]
                        att_index_new.extend([
                            attack_outputs_ind_max,
                            attack_outputs_ind_mid,
                            attack_outputs_ind_min
                        ])
                if last_attack_det_center is not None:
                    last_attack_det_center_ = last_attack_det_center * Ws[0]/W
                    target_center_delta = target_det_center_max - last_attack_det_center_
                    if torch.max(torch.abs(target_center_delta)) > 1:
                        target_center_delta /= torch.max(torch.abs(target_center_delta))
                        target_det_center_max = torch.round(target_det_center_max - target_center_delta).int()
                        target_det_center_mid = torch.round(target_det_center_max / 2).int()
                        target_det_center_min = torch.round(target_det_center_mid / 2).int()
                        target_outputs_ind_max = target_det_center_max[0] + target_det_center_max[1] * Ws[0]
                        target_outputs_ind_mid = target_det_center_mid[0] + target_det_center_mid[1] * Ws[1] \
                                                 + Ws[0] * Hs[0]
                        target_outputs_ind_min = target_det_center_min[0] + target_det_center_min[1] * Ws[2] \
                                                 + Ws[0] * Hs[0] + Ws[1] * Hs[1]
                        att_index_new.extend([
                            target_outputs_ind_max,
                            target_outputs_ind_mid,
                            target_outputs_ind_min
                        ])

                if len(ori_index) and isinstance(ori_index, list):
                    ori_index_re = torch.stack(ori_index[3:] + ori_index[:3]).type(torch.int64)
                    ori_index = torch.stack(ori_index).type(torch.int64)
                if len(att_index_new):
                    att_index_new = torch.stack(att_index_new).type(torch.int64)
                    if len(att_index_new) == 3:
                        if last_target_det_center is None:
                            ori_index_re_ = ori_index_re[3:]
                        else:
                            ori_index_re_ = ori_index_re[:3]
                    else:
                        ori_index_re_ = ori_index_re
                if len(att_index_new):
                    att_index = att_index_new
            loss_att = 0
            loss_ori = 0
            loss_wh = 0
            if len(att_index):
                n_att_index_lst = []
                n_ori_index_lst = []
                for hm_ind in range(len(att_index) // 3):
                    for n_i in range(3):
                        for n_j in range(3):
                            att_hm_ind = att_index[hm_ind * 3].item()
                            att_hm_ind = att_hm_ind + (n_i - 1) * Ws[0] + (n_j - 1)
                            att_hm_ind = max(0, min(Hs[0]*Ws[0]-1, att_hm_ind))
                            n_att_index_lst.append(att_hm_ind)
                            ori_hm_ind = ori_index_re_[hm_ind * 3].item()
                            ori_hm_ind = ori_hm_ind + (n_i - 1) * Ws[0] + (n_j - 1)
                            ori_hm_ind = max(0, min(Hs[0]*Ws[0]-1, ori_hm_ind))
                            n_ori_index_lst.append(ori_hm_ind)
                    n_att_index_lst.append(att_index[hm_ind * 3 + 1].item())
                    n_att_index_lst.append(att_index[hm_ind * 3 + 2].item())
                    n_ori_index_lst.append(ori_index_re_[hm_ind * 3 + 1].item())
                    n_ori_index_lst.append(ori_index_re_[hm_ind * 3 + 2].item())

                loss_att += ((1 - outputs[:, -1][n_att_index_lst]) ** 2 *
                         torch.log(torch.clip(outputs[:, -1][n_att_index_lst], min=1e-4, max=1 - 1e-4))).mean()
                loss_att += ((1 - outputs[:, -2][n_att_index_lst]) ** 2 *
                             torch.log(torch.clip(outputs[:, -2][n_att_index_lst], min=1e-4, max=1 - 1e-4))).mean()
                loss_ori += ((outputs[:, -1][n_ori_index_lst]) ** 2 *
                         torch.log(torch.clip(1-outputs[:, -1][n_ori_index_lst], min=1e-4, max=1 - 1e-4))).mean()
                loss_ori += ((outputs[:, -2][n_ori_index_lst]) ** 2 *
                             torch.log(torch.clip(1 - outputs[:, -2][n_ori_index_lst], min=1e-4, max=1 - 1e-4))).mean()

                outputs_wh = outputs[:, 2:4][n_att_index_lst] - outputs[:, :2][n_att_index_lst]
                loss_wh += -smoothL1(outputs_wh, reg_wh[n_ori_index_lst])

            loss = loss_att + loss_ori + loss_wh * 0.1
            if isinstance(loss, float):
                suc = False
                break
            loss.backward()
            grad = imgs.grad
            grad /= (grad ** 2).sum().sqrt() + 1e-8

            noise += grad

            imgs = (imgs_ori + noise)
            imgs[0, 0] = torch.clip(imgs[0, 0], min=-0.485 / 0.229, max=(1 - 0.485) / 0.229)
            imgs[0, 1] = torch.clip(imgs[0, 1], min=-0.456 / 0.224, max=(1 - 0.456) / 0.224)
            imgs[0, 2] = torch.clip(imgs[0, 2], min=-0.406 / 0.225, max=(1 - 0.406) / 0.225)
            imgs = imgs.data
            outputs, ae_attack_id, ae_target_id, _ = self.forwardFeatureSg(
                imgs,
                img_info,
                dets,
                dets_second,
                attack_id,
                attack_ind,
                target_id,
                target_ind,
                last_info
            )
            if ae_attack_id != attack_id and ae_attack_id is not None:
                break
                # if ae_attack_id == target_id and ae_target_id == attack_id:
                #     break
                # elif ae_attack_id == target_id or ae_target_id == attack_id:
                #     noise_0 = noise.clone()
                #     i_0 = i
                # else:
                #     noise_1 = noise.clone()
                #     i_1 = i

            if i > 60:
                # if noise_0 is not None:
                #     return noise_0, i_0, suc
                # elif noise_1 is not None:
                #     return noise_1, i_1, suc
                suc = False
                break
        return noise, i, suc


    def forwardFeatureSg(
            self,
            imgs,
            img_info,
            dets_,
            dets_second_,
            attack_id,
            attack_ind,
            target_id,
            target_ind,
            last_info
    ):
        width = img_info[1][0].item()
        height = img_info[0][0].item()
        inp_height = imgs.shape[2]
        inp_width = imgs.shape[3]

        imgs.requires_grad = True
        self.model_2.zero_grad()

        outputs = self.model_2(imgs)
        if self.decoder is not None:
            outputs = self.decoder(outputs, dtype=outputs.type())

        outputs_post, outputs_index = postprocess(outputs.detach(), self.num_classes, self.confthre, self.nmsthre)
        output_results = outputs_post[0]
        outputs = outputs[0]

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        scale = min(inp_height / float(height), inp_width / float(width))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        outputs_index_1 = outputs_index[remain_inds]
        outputs_index_2 = outputs_index[inds_second]
        hm_index = torch.cat([outputs_index_1, outputs_index_2])
        # import pdb; pdb.set_trace()

        dets_all_ = np.concatenate([dets_, dets_second_])
        dets_all = np.concatenate([dets, dets_second])

        ious = bbox_ious(np.ascontiguousarray(dets_all_[[attack_ind, target_ind], :4], dtype=np.float64),
                         np.ascontiguousarray(dets_all[:, :4], dtype=np.float64))

        row_inds, col_inds = linear_sum_assignment(-ious)

        ae_attack_id = None
        ae_target_id = None
        # if ious[0, det_ind[0]] < 0.6 or ious[1, det_ind[1]] < 0.6:
        #     return outputs, ae_attack_id, ae_target_id, None

        if row_inds[0] == 0:
            ae_attack_ind = col_inds[0]
            ae_target_ind = col_inds[1]
        else:
            ae_attack_ind = col_inds[1]
            ae_target_ind = col_inds[0]

        # hm_index[[attack_ind, target_ind]] = hm_index[[ae_attack_ind, ae_target_ind]]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = copy.deepcopy(last_info['last_unconfirmed'])
        strack_pool = copy.deepcopy(last_info['last_strack_pool'])

        ''' Step 2: First association, with high score detection boxes'''
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if idet == ae_attack_ind:
                ae_attack_id = track.track_id
            elif idet == ae_target_ind:
                ae_target_id = track.track_id
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if idet + len(dets) == ae_attack_ind:
                ae_attack_id = track.track_id
            elif idet + len(dets) == ae_target_ind:
                ae_target_id = track.track_id

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        for i, idet in enumerate(u_detection):
            if idet == ae_attack_ind:
                ae_attack_ind = i
            elif idet == ae_target_ind:
                ae_target_ind = i
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            if idet == ae_attack_ind:
                ae_attack_id = track.track_id
            elif idet == ae_target_ind:
                ae_target_id = track.track_id

        return outputs, ae_attack_id, ae_target_id, hm_index


    def CheckFit(self, dets, scores_keep, dets_second, scores_second, attack_ids, attack_inds):
        attack_dets = np.concatenate([dets, dets_second])[attack_inds][:4]
        ad_attack_dets = []
        ad_attack_ids = []
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        strack_pool = copy.deepcopy(self.ad_last_info['last_strack_pool'])
        unconfirmed = copy.deepcopy(self.ad_last_info['last_unconfirmed'])

        ''' Step 2: First association, with high score detection boxes'''
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.track_id in attack_ids:
                ad_attack_dets.append(det.tlbr)
                ad_attack_ids.append(track.track_id)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.track_id in attack_ids:
                ad_attack_dets.append(det.tlbr)
                ad_attack_ids.append(track.track_id)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            det = detections[idet]
            if track.track_id in attack_ids:
                ad_attack_dets.append(det.tlbr)
                ad_attack_ids.append(track.track_id)

        if len(ad_attack_dets) == 0:
            return []

        ori_dets = np.array(attack_dets)
        ad_dets = np.array(ad_attack_dets)

        ious = bbox_ious(ori_dets.astype(np.float64), ad_dets.astype(np.float64))
        ious_inds = np.argmax(ious, axis=1)
        attack_index = []
        for i, ind in enumerate(ious_inds):
            if ious[i, ind] > 0.8:
                attack_index.append(i)

        return attack_index


    def update(self, imgs, img_info, img_size, data_list, ids, **kwargs):
        self.frame_id += 1
        self_track_id = kwargs.get('track_id', None)
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        with torch.no_grad():
            outputs = self.model(imgs.data)
        if self.decoder is not None:
            outputs = self.decoder(outputs, dtype=outputs.type())
        outputs, _ = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        # import pdb; pdb.set_trace()
        output_results = self.convert_to_coco_format(outputs, img_info, ids)
        data_list.extend(output_results)
        output_results = outputs[0]

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.detach().cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        id_t_5 = id_d_5 = -1
        id_t_11 = id_d_11 = -1
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.track_id == 5:
                id_t_5 = itracked
                id_d_5 = idet
            elif track.track_id == 11:
                id_t_11 = itracked
                id_d_11 = idet
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # if self.frame_id == 10 and self.args.attack:
        #     print(f'ori:\t5:{dists[4, 6]}\t11:{dists[10, 9]}')
        #     print(f'att:\t5:{dists[4, 9]}\t11:{dists[10, 6]}')
        #     import pdb; pdb.set_trace()
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id, track_id=self_track_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        self.ad_last_info = {
            'last_strack_pool': copy.deepcopy(strack_pool),
            'last_unconfirmed': copy.deepcopy(unconfirmed)
        }

        return output_stracks


    def update_attack_sg(self, imgs, img_info, img_size, data_list, ids, **kwargs):
        self.frame_id_ += 1
        attack_id = kwargs['attack_id']
        self_track_id_ori = kwargs.get('track_id', {}).get('origin', None)
        self_track_id_att = kwargs.get('track_id', {}).get('attack', None)
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        imgs.requires_grad = True
        self.model_2.zero_grad()
        outputs = self.model_2(imgs)

        if self.decoder is not None:
            outputs = self.decoder(outputs, dtype=outputs.type())
        outputs_post, outputs_index = postprocess(outputs.detach(), self.num_classes, self.confthre, self.nmsthre)

        output_results = self.convert_to_coco_format([outputs_post[0].detach()], img_info, ids)
        data_list.extend(output_results)
        output_results = outputs_post[0]
        outputs = outputs[0]

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.detach().cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        outputs_index_1 = outputs_index[remain_inds]
        outputs_index_2 = outputs_index[inds_second]

        dets_ids = [None for _ in range(len(dets)+len(dets_second))]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks_:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks_)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id_)
                activated_starcks.append(track)
            else:
                track.re_activate_(det, self.frame_id_, new_id=False)
                refind_stracks.append(track)
            dets_ids[idet] = track.track_id
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id_)
                activated_starcks.append(track)
            else:
                track.re_activate_(det, self.frame_id_, new_id=False)
                refind_stracks.append(track)
            dets_ids[idet + len(dets)] = track.track_id

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id_)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate_(self.kalman_filter, self.frame_id_, track_id=self_track_id_ori)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks_:
            if self.frame_id_ - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks_ = [t for t in self.tracked_stracks_ if t.state == TrackState.Tracked]
        self.tracked_stracks_ = joint_stracks(self.tracked_stracks_, activated_starcks)
        self.tracked_stracks_ = joint_stracks(self.tracked_stracks_, refind_stracks)
        self.lost_stracks_ = sub_stracks(self.lost_stracks_, self.tracked_stracks_)
        self.lost_stracks_.extend(lost_stracks)
        self.lost_stracks_ = sub_stracks(self.lost_stracks_, self.removed_stracks_)
        self.removed_stracks_.extend(removed_stracks)
        self.tracked_stracks_, self.lost_stracks_ = remove_duplicate_stracks(self.tracked_stracks_, self.lost_stracks_)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks_ if track.is_activated]

        dets_all = np.concatenate([dets, dets_second])
        noise = None
        suc = 0
        for attack_ind, track_id in enumerate(dets_ids):
            if track_id == attack_id:
                if self.args.attack_id >= 0:
                    if not hasattr(self, f'frames_{attack_id}'):
                        setattr(self, f'frames_{attack_id}', 0)
                    if getattr(self, f'frames_{attack_id}') < self.FRAME_THR:
                        setattr(self, f'frames_{attack_id}', getattr(self, f'frames_{attack_id}') + 1)
                        break
                fit = self.CheckFit(dets, scores_keep, dets_second, scores_second, [attack_id], [attack_ind])
                ious = bbox_ious(np.ascontiguousarray(dets_all[:, :4], dtype=np.float64),
                                 np.ascontiguousarray(dets_all[:, :4], dtype=np.float64))

                ious[range(len(ious)), range(len(ious))] = 0
                dis = bbox_dis(np.ascontiguousarray(dets_all[:, :4], dtype=np.float64),
                               np.ascontiguousarray(dets_all[:, :4], dtype=np.float64))
                dis[range(len(dis)), range(len(dis))] = np.inf
                target_ind = np.argmax(ious[attack_ind])
                if ious[attack_ind][target_ind] >= self.attack_iou_thr:
                    if ious[attack_ind][target_ind] == 0:
                        target_ind = np.argmin(dis[attack_ind])
                    target_id = dets_ids[target_ind]
                    if fit:
                        noise, attack_iter, suc = self.ifgsm_adam_sg(
                            imgs,
                            img_info,
                            dets,
                            dets_second,
                            outputs_index_1,
                            outputs_index_2,
                            last_info=self.ad_last_info,
                            outputs_ori=outputs,
                            attack_id=attack_id,
                            attack_ind=attack_ind,
                            target_id=target_id,
                            target_ind=target_ind
                        )
                        self.attack_iou_thr = 0
                        if suc:
                            suc = 1
                            print(
                                f'attack id: {attack_id}\tattack frame {self.frame_id_}: SUCCESS\tl2 distance: {(noise ** 2).sum().sqrt().item()}\titeration: {attack_iter}')
                        else:
                            suc = 2
                            print(
                                f'attack id: {attack_id}\tattack frame {self.frame_id_}: FAIL\tl2 distance: {(noise ** 2).sum().sqrt().item()}\titeration: {attack_iter}')
                    else:
                        suc = 3
                    if ious[attack_ind][target_ind] == 0:
                        self.temp_i += 1
                        if self.temp_i >= 10:
                            self.attack_iou_thr = self.ATTACK_IOU_THR
                    else:
                        self.temp_i = 0
                else:
                    self.attack_iou_thr = self.ATTACK_IOU_THR
                    if fit:
                        suc = 2
                break

        adImg = cv2.imread(os.path.join(self.args.img_dir, img_info[-1][0]))
        if noise is not None:
            l2_dis = (noise ** 2).sum().sqrt().item()

            imgs = (imgs + noise)
            imgs[0, 0] = torch.clip(imgs[0, 0], min=-0.485 / 0.229, max=(1 - 0.485) / 0.229)
            imgs[0, 1] = torch.clip(imgs[0, 1], min=-0.456 / 0.224, max=(1 - 0.456) / 0.224)
            imgs[0, 2] = torch.clip(imgs[0, 2], min=-0.406 / 0.225, max=(1 - 0.406) / 0.225)
            imgs = imgs.data

            noise = self.recoverNoise(noise, adImg)
            adImg = np.clip(adImg + noise, a_min=0, a_max=255)

            noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
            noise = (noise * 255).astype(np.uint8)
        else:
            l2_dis = None
        output_stracks_att = self.update(imgs, img_info, img_size, [], ids, track_id=self_track_id_att)

        return output_stracks_att, adImg, noise, l2_dis, suc

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
