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

mse = torch.nn.MSELoss()

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
            decoder=None
    ):
        self.model = model
        self.model_2 = copy.deepcopy(model)
        self.decoder = decoder
        self.num_classes = num_classes
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.convert_to_coco_format = convert_to_coco_format

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.tracked_stracks_ = []  # type: list[STrack]
        self.lost_stracks_ = []  # type: list[STrack]
        self.removed_stracks_ = []  # type: list[STrack]

        self.frame_id = 0
        self.frame_id_ = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.FRAME_THR = 10
        self.ATTACK_IOU_THR = 0.3
        self.attack_iou_thr = self.ATTACK_IOU_THR

        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda()

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
            target_ind,
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999
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

        adam_m = 0
        adam_v = 0

        i = 0
        j = -1
        suc = True
        while True:
            i += 1
            loss = 0
            loss -= mse(imgs, imgs_ori)

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

            if i in [1, 10, 20, 30, 40, 60, 80]:
                ori_index = []
                att_index = []
                attack_det_center = torch.stack([attack_outputs_ind % Ws[attack_i], attack_outputs_ind // Ws[attack_i]]).float().cuda()
                target_det_center = torch.stack([target_outputs_ind % Ws[target_i], target_outputs_ind // Ws[target_i]]).float().cuda()
                if last_target_det_center is not None:
                    last_target_det_center_ = last_target_det_center * Ws[attack_i]/W
                    attack_center_delta = attack_det_center - last_target_det_center_
                    if torch.max(torch.abs(attack_center_delta)) >= 1:
                        attack_center_delta /= torch.max(torch.abs(attack_center_delta))
                        attack_det_center = torch.round(attack_det_center - attack_center_delta).int()
                        attack_outputs_ind = attack_det_center[0] + attack_det_center[1] * Ws[attack_i]
                        for a_i in range(attack_i):
                            attack_outputs_ind += Ws[a_i] * Hs[a_i]
                        ori_index.append(hm_index[attack_ind].clone())
                        hm_index[attack_ind] = attack_outputs_ind
                        att_index.append(attack_outputs_ind)
                if last_attack_det_center is not None:
                    last_attack_det_center_ = last_attack_det_center * Ws[target_i]/W
                    target_center_delta = target_det_center - last_attack_det_center_
                    if torch.max(torch.abs(target_center_delta)) >= 1:
                        target_center_delta /= torch.max(torch.abs(target_center_delta))
                        target_det_center = torch.round(target_det_center - target_center_delta).int()
                        target_outputs_ind = target_det_center[0] + target_det_center[1] * Ws[target_i]
                        for t_i in range(target_i):
                            target_outputs_ind += Ws[t_i] * Hs[t_i]
                        ori_index.append(hm_index[target_ind].clone())
                        hm_index[target_ind] = target_outputs_ind
                        att_index.append(target_outputs_ind)
                if len(att_index):
                    att_index = torch.stack(att_index).type(torch.int64)
                if len(ori_index):
                    ori_index = torch.stack(ori_index).type(torch.int64)


            # loss += ((1 - outputs[:, -1][hm_index]) ** 2 *
            #         torch.log(outputs[:, -1][hm_index])).mean()
            if len(att_index):
                loss += ((1 - outputs[:, -1][att_index]) ** 2 *
                         torch.log(outputs[:, -1][att_index])).mean()
                # loss += ((1 - outputs[:, -2][att_index]) ** 2 *
                #          torch.log(outputs[:, -2][att_index])).mean()
            if len(ori_index):
                loss += ((outputs[:, -1][ori_index]) ** 2 *
                         torch.log(1-outputs[:, -1][ori_index])).mean()
                # loss += ((outputs[:, -2][ori_index]) ** 2 *
                #          torch.log(1 - outputs[:, -2][ori_index])).mean()

            # loss -= mse(outputs[:, :4][att_index], reg[ori_index])
            # import pdb; pdb.set_trace()
            # loss -= mse(outputs['wh'].view(-1)[hm_index], wh_ori.view(-1)[hm_index_ori])
            # loss -= mse(outputs['reg'].view(-1)[hm_index], reg_ori.view(-1)[hm_index_ori])
            # import pdb; pdb.set_trace()

            loss.backward()
            grad = imgs.grad
            if torch.isnan(grad).sum()>0:
                import pdb; pdb.set_trace()

            # adam_m = beta_1 * adam_m + (1 - beta_1) * grad
            # adam_v = beta_2 * adam_v + (1 - beta_2) * (grad ** 2)
            #
            # adam_m_ = adam_m / (1 - beta_1 ** i)
            # adam_v_ = adam_v / (1 - beta_2 ** i)
            #
            # update_grad = lr * adam_m_ / (adam_v_.sqrt() + 1e-4)

            noise += grad

            imgs = (imgs_ori + noise).data
            imgs[0, 0] = torch.clip(imgs[0, 0], min=-0.485 / 0.229, max=(1 - 0.485) / 0.229)
            imgs[0, 1] = torch.clip(imgs[0, 1], min=-0.456 / 0.224, max=(1 - 0.456) / 0.224)
            imgs[0, 2] = torch.clip(imgs[0, 2], min=-0.406 / 0.225, max=(1 - 0.406) / 0.225)
            print(hm_index_ori[[attack_ind, target_ind]], hm_index[[attack_ind, target_ind]])
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
            # if hm_index is None:
            #     hm_index = hm_index_ori
            if ae_attack_id != attack_id and ae_attack_id is not None:
                break

            # if ae_attack_id == target_id and ae_target_id == attack_id:
            #     break

            if i > 80:
                suc = False
                break
                # return None, -1
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

        outputs_post, outputs_index = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        output_results = outputs_post[0]
        outputs = outputs[0]

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.detach().cpu().numpy()
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

        det_ind = np.argmax(ious, axis=1)
        ae_attack_id = None
        ae_target_id = None
        # if ious[0, det_ind[0]] < 0.6 or ious[1, det_ind[1]] < 0.6:
        #     return outputs, ae_attack_id, ae_target_id, None

        ae_attack_ind = det_ind[0]
        ae_target_ind = det_ind[1]

        hm_index[[attack_ind, target_ind]] = hm_index[[ae_attack_ind, ae_target_ind]]
        print(hm_index[[attack_ind, target_ind]])

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
                return outputs, ae_attack_id, ae_target_id, hm_index
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
                return outputs, ae_attack_id, ae_target_id, hm_index


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        for i, idet in enumerate(u_detection):
            if idet == ae_attack_ind:
                ae_attack_ind = i
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            if idet == ae_attack_ind:
                ae_attack_id = track.track_id
                return outputs, ae_attack_id, ae_target_id, hm_index

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
            if ious[i, ind] > 0.9:
                attack_index.append(i)

        return attack_index


    def update(self, imgs, img_info, img_size, data_list, ids):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        with torch.no_grad():
            outputs = self.model(imgs)
        if self.decoder is not None:
            outputs = self.decoder(outputs, dtype=outputs.type())
        outputs, _ = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        # import pdb; pdb.set_trace()
        output_results = self.convert_to_coco_format([outputs[0].detach()], img_info, ids)
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
            track.activate(self.kalman_filter, self.frame_id)
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


    def update_attack_sg(self, imgs, img_info, img_size, data_list, ids):
        self.frame_id_ += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        imgs.requires_grad = True
        self.model.zero_grad()
        outputs = self.model(imgs)

        if self.decoder is not None:
            outputs = self.decoder(outputs, dtype=outputs.type())
        outputs_post, outputs_index = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

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
            track.activate_(self.kalman_filter, self.frame_id_)
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
        attack_id = 5
        noise = None
        suc = 0
        for attack_ind, track_id in enumerate(dets_ids):
            if track_id == attack_id:
                # if self.opt.attack_id > 0:
                if not hasattr(self, f'frames_{attack_id}'):
                    setattr(self, f'frames_{attack_id}', 0)
                if getattr(self, f'frames_{attack_id}') < self.FRAME_THR:
                    setattr(self, f'frames_{attack_id}', getattr(self, f'frames_{attack_id}') + 1)
                    break
                ious = bbox_ious(np.ascontiguousarray(dets_all[:, :4], dtype=np.float64),
                                 np.ascontiguousarray(dets_all[:, :4], dtype=np.float64))

                ious[range(len(ious)), range(len(ious))] = 0
                target_ind = np.argmax(ious[attack_ind])
                if ious[attack_ind][target_ind] >= self.attack_iou_thr:
                    target_id = dets_ids[target_ind]
                    fit = self.CheckFit(dets, scores_keep, dets_second, scores_second, [attack_id], [attack_ind])
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
                break

        # if noise is not None:
        #     l2_dis = (noise ** 2).sum().sqrt().item()
        #     im_blob = torch.clip(im_blob + noise, min=0, max=1)
        #
        #     noise = self.recoverNoise(noise, img0)
        #     adImg = np.clip(img0 + noise, a_min=0, a_max=255)
        #
        #     noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        #     noise = (noise * 255).astype(np.uint8)
        # else:
        #     l2_dis = None
        #     adImg = img0
        output_stracks_att = self.update(imgs, img_info, img_size, [], ids)

        return output_stracks

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
