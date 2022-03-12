from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.tracker.byte_tracker import BYTETracker
from yolox.sort_tracker.sort import Sort
from yolox.deepsort_tracker.deepsort import DeepSort
from yolox.motdt_tracker.motdt_tracker import OnlineTracker

from yolox.tracking_utils import visualization as vis
from yolox.tracker.basetrack import BaseTrack
from scipy.optimize import linear_sum_assignment


import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import cv2
import copy
import numpy as np

from cython_bbox import bbox_overlaps as bbox_ious
class MultipleEval:
    def __init__(self, start_frame, iou_thr):
        self.start_frame = start_frame
        self.iou_thr = iou_thr

    @staticmethod
    def read_result(path):
        f = open(path)
        lines = f.readlines()
        frame2id = {}
        id2frame = {}
        for line in lines:
            frame, id = map(int, line.strip('\n').split(',')[:2])
            bbox = list(map(float, line.strip('\n').split(',')[2:-4]))
            frame2id.setdefault(frame, {})
            id2frame.setdefault(id, {})
            frame2id[frame][id] = bbox
            id2frame[id][frame] = bbox
        return frame2id, id2frame

    @staticmethod
    def tracks_pari(origin_frame2id, attack_frame2id, valid_id2frame):
        tracks_pair_dic = {}
        for id, info in valid_id2frame.items():
            tracks_pair_dic[id] = dict((frame_id, -1) for frame_id in info['frames'])

        for frame_id, frame_info in origin_frame2id.items():
            origin_bbox_info = [[id, bbox] for id, bbox in frame_info.items()]
            origin_bbox = np.array([info[1] for info in origin_bbox_info])
            origin_id = [info[0] for info in origin_bbox_info]
            attack_bbox_info = [[id, bbox] for id, bbox in attack_frame2id[frame_id].items()]
            attack_bbox = np.array([info[1] for info in attack_bbox_info])
            attack_id = [info[0] for info in attack_bbox_info]
            origin_bbox[:, 2:] = origin_bbox[:, 2:] + origin_bbox[:, :2]
            attack_bbox[:, 2:] = attack_bbox[:, 2:] + attack_bbox[:, :2]
            iou = bbox_ious(origin_bbox, attack_bbox)
            origin_inds, attack_inds = linear_sum_assignment(1 - iou)

            for origin_ind, attack_ind in zip(origin_inds, attack_inds):
                if origin_id[origin_ind] in valid_id2frame and iou[origin_ind, attack_ind] > 0.5:
                    tracks_pair_dic[origin_id[origin_ind]][frame_id] = attack_id[attack_ind]
                else:
                    continue
        return tracks_pair_dic

    def get_valid_ids(self, frame2id, id2frame):
        eval_id = []
        valid_id2frame = {}
        for id, frame in id2frame.items():
            if len(frame) > self.start_frame:
                eval_id.append(id)
                valid_frames = list(id2frame[id].keys())
                valid_frames.sort()
                for frame in valid_frames[10:]:
                    if self.eval_frame(frame2id, frame, id):
                        if id not in valid_id2frame:
                            valid_id2frame[id] = {}
                            valid_id2frame[id]['frame2bbox'] = id2frame[id]
                            valid_id2frame[id]['frames'] = list(id2frame[id].keys())
                            valid_id2frame[id]['intersect_frames'] = [frame]
                        else:
                            valid_id2frame[id]['intersect_frames'].append(frame)

        return valid_id2frame

    def eval_frame(self, frame2id, frame_id, persion_id):
        bbox = frame2id[frame_id][persion_id]
        bbox = np.array([bbox])
        bbox[:, 2:] = bbox[:, 2:] + bbox[:, :2]
        comp_bbox = np.array([bbox for id, bbox in frame2id[frame_id].items() if id != persion_id])

        if len(comp_bbox) == 0:
            return False

        comp_bbox[:, 2:] = comp_bbox[:, 2:] + comp_bbox[:, :2]
        ious = bbox_ious(bbox, comp_bbox)

        if (ious > self.iou_thr).any():
            return True
        return False

    def __call__(self, origin_path, attack_path):
        origin_frame2id, origin_id2frame = self.read_result(origin_path)
        attack_frame2id, attack_id2frame = self.read_result(attack_path)

        valid_id2frame = self.get_valid_ids(origin_frame2id, origin_id2frame)
        valid_id_track_pari = self.tracks_pari(origin_frame2id, attack_frame2id, valid_id2frame)

        success_attack = 0
        success_attack_id = set([])
        all_attack_id = set(valid_id_track_pari.keys())
        for id, track_info in valid_id_track_pari.items():
            track_id = [pre_track_id for frame_id, pre_track_id in track_info.items()]
            while -1 in track_id:
                track_id.remove(-1)
            if len(track_id) < self.start_frame:
                continue
            origin_id = track_id[self.start_frame - 1]
            final_index = len(track_id) - 1 - track_id[::-1].index(origin_id)
            if final_index + 1 < len(track_id):
                success_attack += 1
                success_attack_id.add(id)

        return success_attack_id, all_attack_id
class Logger:
    def __init__(self, file):
        self.file = file

    def __call__(self, s):
        print(s)
        print(s, file=self.file)
def write_results_mt(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))
def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """



        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        results_att = []
        results_att_sg = {}
        l2_distance = []
        l2_distance_sg = {}
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter
        attack_frames=0
        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            # x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            # model(x)
            model = model_trt
            
        # tracker = BYTETracker(self.args, model=model, decoder=decoder)
        tracker = None
        ori_thresh = self.args.track_thresh
        last_vdo = None
        vdos = 0
        os.makedirs(self.args.output_dir, exist_ok=True)
        model2 = copy.deepcopy(model)
        tracker = BYTETracker(self.args, self.num_classes, self.confthre, self.nmsthre, self.convert_to_coco_format, model=model,model2=model2,decoder=decoder)
                
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            self.dataloader
        ):
            sg_track_outputs = {}
            # with torch.no_grad():
            # init tracker
            frame_id = info_imgs[2].item()
            video_id = info_imgs[3].item()
            img_file_name = info_imgs[4]
            video_name = img_file_name[0].split('/')[0]
            if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                self.args.track_buffer = 14
            elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                self.args.track_buffer = 25
            else:
                self.args.track_buffer = 30

            if video_name == 'MOT17-01-FRCNN':
                self.args.track_thresh = 0.65
            elif video_name == 'MOT17-06-FRCNN':
                self.args.track_thresh = 0.65
            elif video_name == 'MOT17-12-FRCNN':
                self.args.track_thresh = 0.7
            elif video_name == 'MOT17-14-FRCNN':
                self.args.track_thresh = 0.67
            else:
                self.args.track_thresh = ori_thresh

            if video_name == 'MOT20-06' or video_name == 'MOT20-08':
                self.args.track_thresh = 0.3
            else:
                self.args.track_thresh = ori_thresh

            if video_name not in video_names:
                video_names[video_id] = video_name
            if frame_id == 1:
                vdos += 1
                if last_vdo and self.args.attack == 'single' and self.args.attack_id == -1 and self.args.method in ['ids', 'dets', 'feat', 'cl', 'hijack']:
                    output_file = os.path.join(self.args.output_dir, f'{last_vdo}_attack_result.txt')
                    file = open(output_file, 'w')
                    out_logger = Logger(file)
                    keys = list(suc_frequency_ids.keys())
                    for key in keys:
                        if suc_frequency_ids[key] == 0:
                            del suc_frequency_ids[key]
                    suc_attacked_ids.update(set(suc_frequency_ids.keys()))
                    out_logger('@' * 50 + ' single attack accuracy ' + '@' * 50)
                    out_logger(f'All attacked ids is {need_attack_ids}')
                    out_logger(f'All successfully attacked ids is {suc_attacked_ids}')
                    out_logger(f'All unsuccessfully attacked ids is {need_attack_ids - suc_attacked_ids}')
                    out_logger(
                        f'The accuracy is {round(100 * len(suc_attacked_ids) / len(need_attack_ids), 2) if len(need_attack_ids) else 0}%')
                    out_logger(
                        f'The attacked frames: {sg_attack_frames}\tmin: {min(sg_attack_frames.values()) if len(need_attack_ids) else None}\t'
                        f'max: {max(sg_attack_frames.values()) if len(need_attack_ids) else None}\tmean: {sum(sg_attack_frames.values()) / len(sg_attack_frames) if len(need_attack_ids) else None}')
                    out_logger(
                        f'The mean L2 distance: {dict(zip(suc_attacked_ids, [sum(l2_distance_sg[k]) / max(1e-8, len(l2_distance_sg[k])) for k in suc_attacked_ids])) if len(suc_attacked_ids) else None}')
                    file.close()
                last_vdo_=last_vdo
                last_vdo = video_name
                
                tracker = BYTETracker(self.args, self.num_classes, self.confthre, self.nmsthre, self.convert_to_coco_format, model=model,model2=model2,decoder=decoder)
                
                if len(results) != 0:
                    result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                    if self.args.attack == "multiple":
                        write_results_mt(result_filename, results)
                    else:
                        write_results(result_filename, results)
                if len(results_att) != 0:
                    result_filename = os.path.join(result_folder, '{}_attack_result.txt'.format(video_names[video_id - 1]))
                    if self.args.attack == "multiple":
                        write_results_mt(result_filename, results_att)
                    # else:
                    #     write_results(result_filename, results_att)
                if last_vdo_ and self.args.attack == 'multiple':
                    result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                    output_file = os.path.join(self.args.output_dir, f'{last_vdo}_attack_result.txt')
                    file = open(output_file, 'w')
                    out_logger = Logger(file)
                    eval_attack = MultipleEval(tracker.FRAME_THR, tracker.ATTACK_IOU_THR)
                    suc_attacked_ids, need_attack_ids = eval_attack(result_filename,
                                                                    result_filename.replace('.txt', f'_attack_result.txt'))
                    out_logger('@' * 50 + ' multiple attack accuracy ' + '@' * 50)
                    out_logger(f'All attacked ids is {need_attack_ids}')
                    out_logger(f'All successfully attacked ids is {suc_attacked_ids}')
                    out_logger(f'All unsuccessfully attacked ids is {need_attack_ids - suc_attacked_ids}')
                    out_logger(
                        f'The accuracy is {round(100 * len(suc_attacked_ids) / len(need_attack_ids), 2) if len(need_attack_ids) else None}% | '
                        f'{len(suc_attacked_ids)}/{len(need_attack_ids)}')
                    out_logger(f'The attacked frames: {attack_frames}')
                    #total_attack_frame.append(attack_frames / frame_id)
                    out_logger(f'The mean L2 distance: {sum(l2_distance) / len(l2_distance) if len(l2_distance) else None}')
                    file.close()
                if len(results) != 0:
                    results = []
                if len(results_att) != 0:
                    results_att = []
                BaseTrack.init()
                need_attack_ids = set([])
                suc_attacked_ids = set([])
                frequency_ids = {}
                att_frequency_ids = {}
                trackers_dic = {}
                suc_frequency_ids = {}

                tracked_stracks = []
                lost_stracks = []
                removed_stracks = []
                ad_last_info = {}

                track_id = {'track_id': 1}
                sg_track_ids = {}
                sg_attack_frames = {}
                attack_frames = 0

            imgs = imgs.type(tensor_type)

            # skip the the last iters since batchsize might be not enough for batch inference
            is_time_record = cur_iter < len(self.dataloader) - 1
            if is_time_record:
                start = time.time()
            if frame_id % 20 == 0:
                print(vdos, video_name, frame_id)
            #     outputs = model(imgs)
            #     if decoder is not None:
            #         outputs = decoder(outputs, dtype=outputs.type())
            #
            #     outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            #
            #     if is_time_record:
            #         infer_end = time_synchronized()
            #         inference_time += infer_end - start
            #
            # output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            # data_list.extend(output_results)

            # run tracking
            # img0 = cv2.imread(os.path.join(self.args.img_dir, info_imgs[-1][0]))
            if self.args.attack:
                if self.args.attack == 'single' and self.args.attack_id == -1 and self.args.method in ['ids', 'dets', 'feat', 'cl', 'hijack']:
                    online_targets = tracker.update(imgs, info_imgs, self.img_size, data_list, ids, track_id=track_id)
                    dets = []
                    ids_single = []
                    for strack in online_targets:
                        if strack.track_id not in frequency_ids:
                            frequency_ids[strack.track_id] = 0
                        frequency_ids[strack.track_id] += 1
                        ids_single.append(strack.track_id)
                        dets.append(strack.curr_tlbr.reshape(1, -1))
                    if len(ids_single) > 0:
                        dets = np.concatenate(dets).astype(np.float64)
                        ious = bbox_ious(dets, dets)
                        ious[range(len(dets)), range(len(dets))] = 0
                        for i in range(len(dets)):
                            if (ious[i] > tracker.ATTACK_IOU_THR).sum() > 0 and frequency_ids[ids_single[i]] > tracker.FRAME_THR:
                                need_attack_ids.add(ids_single[i])
                    # for strack in online_targets:
                    #     if strack.track_id not in frequency_ids:
                    #         frequency_ids[strack.track_id] = 0
                    #     frequency_ids[strack.track_id] += 1
                    #     if frequency_ids[strack.track_id] > tracker.FRAME_THR:
                    #         ids_single.append(strack.track_id)
                    #         dets.append(strack.curr_tlbr.reshape(1, -1))
                    # if len(ids_single) > 0:
                    #     dets = np.concatenate(dets).astype(np.float64)
                    #     ious = bbox_ious(dets, dets)
                    #
                    #     ious[range(len(dets)), range(len(dets))] = 0
                    #     for i in range(len(dets)):
                    #         for j in range(len(dets)):
                    #             if ious[i, j] > tracker.ATTACK_IOU_THR:
                    #                 need_attack_ids.add(ids_single[i])

                    for attack_id in need_attack_ids:
                        if attack_id in suc_attacked_ids:
                            continue
                        if self.args.rand:
                            if attack_id not in att_frequency_ids:
                                att_frequency_ids[attack_id] = 0
                            att_frequency_ids[attack_id] += 1
                            if att_frequency_ids[attack_id] > 30:
                                continue
                        if attack_id not in trackers_dic:
                            trackers_dic[attack_id] = BYTETracker(
                                self.args,
                                self.num_classes,
                                self.confthre,
                                self.nmsthre,
                                self.convert_to_coco_format,
                                model=model,
                                model2=model2,
                                decoder=decoder,
                                tracked_stracks=tracked_stracks,
                                lost_stracks=lost_stracks,
                                removed_stracks=removed_stracks,
                                frame_id=frame_id,
                                ad_last_info=ad_last_info
                            )
                            sg_track_ids[attack_id] = {
                                'origin': {'track_id': track_id['track_id']},
                                'attack': {'track_id': track_id['track_id']}
                            }
                        if self.args.method == 'ids':
                            output_stracks_att, adImg, noise, l2_dis, suc = trackers_dic[attack_id].update_attack_sg(
                                imgs,
                                info_imgs,
                                self.img_size,
                                data_list,
                                ids,
                                attack_id=attack_id,
                                track_id=sg_track_ids[attack_id]
                            )
                        elif self.args.method == 'dets':
                            output_stracks_att, adImg, noise, l2_dis, suc = trackers_dic[attack_id].update_attack_sg_det(
                                imgs,
                                info_imgs,
                                self.img_size,
                                data_list,
                                ids,
                                attack_id=attack_id,
                                track_id=sg_track_ids[attack_id]
                            )
                        elif self.args.method == 'hijack':
                            output_stracks_att, adImg, noise, l2_dis, suc = trackers_dic[attack_id].update_attack_sg_hj(
                                imgs,
                                info_imgs,
                                self.img_size,
                                data_list,
                                ids,
                                attack_id=attack_id,
                                track_id=sg_track_ids[attack_id]
                            )

                        sg_track_outputs[attack_id] = {}
                        sg_track_outputs[attack_id]['output_stracks_att'] = output_stracks_att
                        sg_track_outputs[attack_id]['adImg'] = adImg
                        sg_track_outputs[attack_id]['noise'] = noise
                        # print(suc)
                        if suc in [1, 2]:
                            if attack_id not in sg_attack_frames:
                                sg_attack_frames[attack_id] = 0
                            sg_attack_frames[attack_id] += 1
                        if attack_id not in results_att_sg:
                            results_att_sg[attack_id] = []
                        if attack_id not in l2_distance_sg:
                            l2_distance_sg[attack_id] = []
                        if l2_dis is not None:
                            l2_distance_sg[attack_id].append(l2_dis)
                        if suc == 1:
                            suc_frequency_ids[attack_id] = 0
                        elif suc == 2:
                            suc_frequency_ids.pop(attack_id, None)
                        elif suc == 3:
                            if attack_id not in suc_frequency_ids:
                                suc_frequency_ids[attack_id] = 0
                            suc_frequency_ids[attack_id] += 1
                        elif attack_id in suc_frequency_ids:
                            suc_frequency_ids[attack_id] += 1
                            if suc_frequency_ids[attack_id] > 20:
                                suc_attacked_ids.add(attack_id)
                                del trackers_dic[attack_id]
                                torch.cuda.empty_cache()

                    tracked_stracks = copy.deepcopy(tracker.tracked_stracks)
                    lost_stracks = copy.deepcopy(tracker.lost_stracks)
                    removed_stracks = copy.deepcopy(tracker.removed_stracks)
                    ad_last_info = copy.deepcopy(tracker.ad_last_info)
                elif self.args.attack == 'single' and self.args.method == "ids":
                    online_targets, adImg, noise, l2_dis, suc = tracker.update_attack_sg(imgs, info_imgs, self.img_size, data_list, ids, attack_id=self.args.attack_id)
                elif self.args.attack == "multiple":
                    online_targets_ori,output_stracks_att, adImg, noise, l2_dis=tracker.update_attack_mt(imgs, info_imgs, self.img_size, data_list, ids, attack_id=None)
                    if l2_dis is not None:
                        l2_distance.append(l2_dis)
                        attack_frames += 1
                
                elif self.args.attack == "multiple":
                    online_tlwhs_att = []
                    online_ids_att = []
                    for t in output_stracks_att:
                        # tlwh = t.tlwh
                        tlwh = t.tlbr_to_tlwh(t.curr_tlbr)
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > 1.6
                        if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                            online_tlwhs_att.append(tlwh)
                            online_ids_att.append(tid)
                    results_att.append((frame_id, online_tlwhs_att, online_ids_att))
                #for frame_id, tlwhs, track_ids in results:

            else:
                online_targets = tracker.update(imgs, info_imgs, self.img_size, data_list, ids)
            if self.args.attack == "multiple":
                online_tlwhs = []
                online_ids = []
                for t in online_targets_ori:
                    # tlwh = t.tlwh
                    tlwh = t.tlbr_to_tlwh(t.curr_tlbr)
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                    # if t.exist_len > 10:
                    #     all_effective_ids.add(t.track_id)
                results.append((frame_id, online_tlwhs, online_ids))
            
                

            if is_time_record:
                infer_end = time_synchronized()
                inference_time += infer_end - start
            # if self.args.attack == 'single' and self.args.attack_id == -1:
            #     for key in sg_track_outputs.keys():
            #         # cv2.imwrite(imgPath.replace('.jpg', f'_{key}.jpg'), sg_track_outputs[key]['adImg'])
            #         # if sg_track_outputs[key]['noise'] is not None:
            #         #     cv2.imwrite(noisePath.replace('.jpg', f'_{key}.jpg'), sg_track_outputs[key]['noise'])
            #         online_tlwhs_att = []
            #         online_ids_att = []
            #         for t in sg_track_outputs[key]['output_stracks_att']:
            #             # tlwh = t.tlwh
            #             tlwh = t.tlbr_to_tlwh(t.curr_tlbr)
            #             tid = t.track_id
            #             vertical = tlwh[2] / tlwh[3] > 1.6
            #             if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
            #                 online_tlwhs_att.append(tlwh)
            #                 online_ids_att.append(tid)
            #         results_att_sg[key].append((frame_id + 1, online_tlwhs_att, online_ids_att))
            #         sg_track_outputs[key]['online_tlwhs_att'] = online_tlwhs_att
            #         sg_track_outputs[key]['online_ids_att'] = online_ids_att

            # online_tlwhs = []
            # online_ids = []
            # online_scores = []
            # for t in online_targets:
            #     tlwh = t.tlbr_to_tlwh(t.curr_tlbr)
            #     tid = t.track_id
            #     vertical = tlwh[2] / tlwh[3] > 1.6
            #     if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
            #         online_tlwhs.append(tlwh)
            #         online_ids.append(tid)
            #         online_scores.append(t.score)
            # save results
            # results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            # if cur_iter == len(self.dataloader) - 1:
            #     result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
            #     write_results(result_filename, results)
            # if self.args.attack == 'single' and self.args.attack_id == -1:
            #     for key in sg_track_outputs.keys():
            #         img0 = sg_track_outputs[key]['adImg'].astype(np.uint8)
            #         sg_track_outputs[key]['online_im'] = vis.plot_tracking(
            #             img0,
            #             sg_track_outputs[key]['online_tlwhs_att'],
            #             sg_track_outputs[key]['online_ids_att'],
            #             frame_id=frame_id,
            #             fps=0
            #         )
            # online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
            #                           fps=0)
            # if self.args.attack:
            #     save_dir = f'/home/derry/Disk/data/B_MOT/att/{video_name}'
            # else:
            #     save_dir = f'/home/derry/Disk/data/B_MOT/ori/{video_name}'
            # os.makedirs(save_dir, exist_ok=True)
            # if self.args.attack == 'single' and self.args.attack_id == -1:
            #     for key in sg_track_outputs.keys():
            #         cv2.imwrite(os.path.join(save_dir, '{:05d}_{}.jpg'.format(frame_id, key)),
            #                     sg_track_outputs[key]['online_im'])
            # cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        if self.args.attack == "multiple":
            result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
            write_results_mt(result_filename, results)
            print("a"*100)
            print(results_att)
            print("a"*100)
            write_results_mt(result_filename.replace('.txt', '_attack.txt'), results_att)


        if last_vdo and self.args.attack == 'single' and self.args.attack_id == -1:
            output_file = os.path.join(self.args.output_dir, f'{last_vdo}_attack_result.txt')
            file = open(output_file, 'w')
            out_logger = Logger(file)
            keys = list(suc_frequency_ids.keys())
            for key in keys:
                if suc_frequency_ids[key] == 0:
                    del suc_frequency_ids[key]
            suc_attacked_ids.update(set(suc_frequency_ids.keys()))
            out_logger('@' * 50 + ' single attack accuracy ' + '@' * 50)
            out_logger(f'All attacked ids is {need_attack_ids}')
            out_logger(f'All successfully attacked ids is {suc_attacked_ids}')
            out_logger(f'All unsuccessfully attacked ids is {need_attack_ids - suc_attacked_ids}')
            out_logger(
                f'The accuracy is {round(100 * len(suc_attacked_ids) / len(need_attack_ids), 2) if len(need_attack_ids) else 0}%')
            out_logger(
                f'The mean L2 distance: {dict(zip(suc_attacked_ids, [sum(l2_distance_sg[k]) / max(1e-8, len(l2_distance_sg[k])) for k in suc_attacked_ids])) if len(suc_attacked_ids) else None}')
            out_logger(
                f'The attacked frames: {sg_attack_frames}\tmin: {min(sg_attack_frames.values()) if len(need_attack_ids) else None}\t'
                f'max: {max(sg_attack_frames.values()) if len(need_attack_ids) else None}\tmean: {sum(sg_attack_frames.values()) / len(sg_attack_frames) if len(need_attack_ids) else None}')
            
            file.close()
        elif last_vdo and self.args.attack == 'multiple':
            output_file = os.path.join(self.args.output_dir, f'{last_vdo}_attack_result.txt')
            file = open(output_file, 'w')
            out_logger = Logger(file)
            result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))

            eval_attack = MultipleEval(tracker.FRAME_THR, tracker.ATTACK_IOU_THR)
            suc_attacked_ids, need_attack_ids = eval_attack(result_filename,
                                                            result_filename.replace('.txt', f'_attack.txt'))
            out_logger('@' * 50 + ' multiple attack accuracy ' + '@' * 50)
            out_logger(f'All attacked ids is {need_attack_ids}')
            out_logger(f'All successfully attacked ids is {suc_attacked_ids}')
            out_logger(f'All unsuccessfully attacked ids is {need_attack_ids - suc_attacked_ids}')
            out_logger(
                f'The accuracy is {round(100 * len(suc_attacked_ids) / len(need_attack_ids), 2) if len(need_attack_ids) else None}% | '
                f'{len(suc_attacked_ids)}/{len(need_attack_ids)}')
            out_logger(f'The attacked frames: {attack_frames}')
            total_attack_frame.append(attack_frames / frame_id)
            out_logger(f'The mean L2 distance: {sum(l2_distance) / len(l2_distance) if len(l2_distance) else None}')
            file.close()
        raise RuntimeError('Finish')
        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_sort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = Sort(self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = Sort(self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_deepsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_motdt(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            '''
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            #from pycocotools.cocoeval import COCOeval
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
