# Copyright (c) Tencent Inc. All rights reserved.
import sys

sys.path.append('/data/yolo-uniow')
import os
import cv2
import time
import json
from tqdm import tqdm
import numpy as np

import torch
from mmengine.config import Config
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

import supervision as sv
import logging
from visualize import visualize_func
from bytetrack import ByteTrack
from mmdet.evaluation import bbox_overlaps


def nms(pred_instances, iou_threshold=0.8):
    # 如果预测结果为空，直接返回
    if len(pred_instances) == 0:
        return pred_instances

    # 提取数据
    boxes = pred_instances.bboxes
    labels = pred_instances.labels

    # 初始化保留索引列表
    keep = [True] * len(boxes)

    # 应用NMS
    for i in range(len(boxes)):
        if not keep[i]:
            continue

        # 只与后面的框比较（因为已经按分数排序）
        for j in range(i + 1, len(boxes)):
            if not keep[j]:
                continue

            # 只比较相同类别的框
            if labels[i] == labels[j]:
                # 计算IoU
                iou = bbox_overlaps(boxes[i:i + 1], boxes[j:j + 1])[0, 0]
                if iou > iou_threshold:
                    # 移除分数较低的框
                    keep[j] = False

    # 创建新的实例
    keep_indices = torch.tensor([i for i, k in enumerate(keep) if k])
    filtered_instances = pred_instances[keep_indices].clone()

    return filtered_instances


class OVDInfer:
    def __init__(self, cfg, use_tracker=False):
        self.checkpoint = cfg["model_path"]
        self.model_cfg_path = cfg["model_cfg_path"]
        self.model_cfg = Config.fromfile(self.model_cfg_path)
        self.device = cfg.get("device", "cuda:0")
        self.model = init_detector(self.model_cfg, checkpoint=self.checkpoint,
                                   device=self.device)
        # init test pipeline
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=self.model_cfg)
        # test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline_cfg)
        self.text_feats_database = dict()
        self.score_thr = cfg["score_thr"]
        self.width_threshold = cfg.get("width_threshold", None)
        self.with_nms = cfg.get("with_nms", False)

        self.tracker = None
        self.tracker_cfg = cfg.get("tracker", {})
        print("tracker parms: ", self.tracker_cfg)
        self.use_tracker = use_tracker
        if use_tracker:
            self.init_tracker()

        logging.info("OVDInfer初始化已完成！")

    def init_tracker(self):
        self.tracker = ByteTrack(**self.tracker_cfg)
        self.tracker.reset()
        self.use_tracker = True
        logging.info("初始化跟踪器已完成！")

    def reparameterize(self, text, text_key):
        self.model.reparameterize([text])
        self.text_feats_database[text_key] = self.model.text_feats
        logging.info(f"增加{text_key}特征")

    def infer(self, img, texts_lst):
        texts_key = ",".join(texts_lst)
        if texts_key not in self.text_feats_database:
            self.reparameterize(texts_lst, texts_key)
        else:
            text_feats = self.text_feats_database[texts_key]
            self.model.text_feats = text_feats

        data_info = dict(img=img)
        data_info = self.test_pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                          data_samples=[data_info['data_samples']])

        with autocast(enabled=False), torch.no_grad():
            output = self.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances
            pred_instances = pred_instances[pred_instances.scores.float() >
                                            self.score_thr]

        pred_instances = pred_instances.cpu().numpy()

        # 过滤尺度过小的检测框
        if self.width_threshold is not None:
            widths = pred_instances['bboxes'][:, 2] - pred_instances['bboxes'][:, 0]
            keep_indices = widths >= self.width_threshold
            pred_instances = pred_instances[keep_indices]

        if self.with_nms:
            pred_instances = nms(pred_instances)

        pred_instances = sv.Detections(
            xyxy=pred_instances['bboxes'],
            confidence=pred_instances['scores'],
            class_id=pred_instances['labels']
        )
        # Apply tracking if tracker is set
        if self.tracker is not None:
            pred_instances = self.tracker.update_with_detections(
                detections=pred_instances)
            # print(f"predinstances 数量：{len(pred_instances)}, ids: {pred_instances.tracker_id}")

        return pred_instances


    def process_video(self, video_path, texts_lst, output_path, use_tracker,
                      frame_interval=1, save_img=False):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)/frame_interval
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"处理视频: {video_path}")
        print(f"总帧数: {total_frames}, FPS: {fps}")

        infer_times = []
        for i in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            if i % frame_interval != 0:
                continue
            t0 = time.time()
            pred_instances = self.infer(frame, texts_lst)
            infer_times.append(time.time()-t0)

            save_img_folder = os.path.join(
                os.path.dirname(output_path),
                os.path.basename(output_path).split(".")[0])
            os.makedirs(save_img_folder, exist_ok=True)
            img_path = None
            if save_img:
                img_path = os.path.join(save_img_folder, f"{i}.jpg")
            annotated_frame = visualize_func(pred_instances, texts_lst, frame,
                                        use_tracker=use_tracker, save_path=img_path)
            out.write(annotated_frame)

        cap.release()
        out.release()

        print(f"单帧耗时：{np.round(np.mean(infer_times)*1000, 2)} ms")
        print(f"视频处理完成，已保存至: {output_path}")


def test_track():

    cfg = {
        "model_cfg_path": "./configs/yolo_uniow_s_notext-transform.py",
        "model_path": "./models/model.pth",
        # "score_thr": 0.01,
        "score_thr": 0.15,
        "device":"cuda:0",
        "width_threshold": 20,
        "with_nms": True,
        "tracker": {
            "track_thresh": 0.2,
            "init_thresh": 0.15,
            "lost_track_buffer": 5,
            "first_matching_threshold": 0.8,
            "second_matching_threshold": 0.8,
            "minimum_consecutive_frames": 5,
            "std_weight_position": 1.0,
            "std_weight_velocity": 0.01,
            "project_rate": 0.01,
            "u_track_with_all_u_dets": True,
            "use_width_match_adaptive": True,
            "use_width_fuse": True,
            "adaptive_width_ratio": 1.3,
            "width_fuse_area_ratio": 2.0,
            "width_fuse_aspect_ratio": 1.5,
            "one_track_width_ratio": 1.8,
            "one_track_widthfuse_ratio": 1.5
        },
    }
    frame_interval = 1
    use_tracker = True

    ovd_infer = OVDInfer(cfg, use_tracker=use_tracker)

    output = f"./demo_outputs/"
    print("save root: ", output)
    os.makedirs(output, exist_ok=True)

    video_text_pairs = {
        "./samples/park_2person.mp4":["person"],
    }
    if not os.path.exists(output):
        os.makedirs(output)
    for in_path, texts in video_text_pairs.items():
        for text in texts:
            ovd_infer.init_tracker()
            text = [text]
            keytext = "_".join([t.replace(" ", "") for t in text])
            suffix = in_path.split('.')[-1].lower()

            output_path = os.path.join(
                output,
                f"result_{os.path.basename(in_path).split('.')[0]}_{keytext}.{suffix}")
            print(f"output_path: ", output_path)

            ovd_infer.process_video(in_path, text, output_path, use_tracker,
                                    frame_interval=frame_interval, save_img=True)


if __name__ == '__main__':
    test_track()

