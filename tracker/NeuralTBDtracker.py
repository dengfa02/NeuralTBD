import itertools
import os
import os.path as osp
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models import *

from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *

from tracker import matching

from .basetrack import BaseTrack, TrackState

from .cmc import CMCComputer
from .gmc import GMC
from .embedding import EmbeddingComputer


class STrack(BaseTrack):
    # shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat=None, conds_len=10, buffer_size=30):
        self.conds_len = conds_len
        # wait activate
        self.xywh_omemory = deque([], maxlen=buffer_size)  # 存储纯观测原始检测框
        self.xywh_pmemory = deque([], maxlen=buffer_size)  # 由扩散模型预测出来的框
        self.xywh_amemory = deque([], maxlen=buffer_size)  # 匹配后的最终平滑框

        # self.v_prior_memory = deque([], maxlen=conds_len)  # 先验预测速度
        # self.trans_memory = deque([], maxlen=conds_len)  # Transformer 深层特征
        # self.cond_encoded = deque([], maxlen=conds_len)  # 编码后的历史条件特征
        self.conds = deque([], maxlen=conds_len)
        self.vel_mask = deque([False] * conds_len, maxlen=conds_len)  # 初始化全 False表示无mask

        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

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
    def warp_bboxes(bboxes_cxcywh, warp_matrix):
        """
        🚀 向量化版本的仿射变换
        输入 bboxes_cxcywh shape: (N, 4) 或 (4,)
        """
        bboxes = np.atleast_2d(bboxes_cxcywh).astype(np.float32)
        if len(bboxes) == 0:
            return bboxes

        # 1. 批量变换中心点 (N, 2)
        # 构造齐次坐标: [cx, cy, 1] -> shape (N, 3)
        pts = np.hstack([bboxes[:, :2], np.ones((len(bboxes), 1), dtype=np.float32)])

        # 矩阵乘法: (N, 3) @ (3, 2) -> (N, 2)
        # warp_matrix 形状是 (2, 3)，所以需要转置 .T
        new_pts = pts @ warp_matrix.T

        # 2. 批量变换宽高
        scale_w = np.sqrt(warp_matrix[0, 0] ** 2 + warp_matrix[0, 1] ** 2)
        scale_h = np.sqrt(warp_matrix[1, 0] ** 2 + warp_matrix[1, 1] ** 2)

        # 广播机制相乘: (N, 2) * (2,) -> (N, 2)
        new_wh = bboxes[:, 2:4] * np.array([scale_w, scale_h], dtype=np.float32)

        # 拼接返回 (N, 4)
        res = np.hstack([new_pts, new_wh])

        # 如果输入是 1D，返回也保持 1D
        return res[0] if bboxes_cxcywh.ndim == 1 else res

    def apply_camera_motion(self, warp_matrix):
        """
        🚀 O(1) 复杂度的向量化 GMC 历史对齐！消灭所有 Python for 循环
        """
        # 1. 扭曲当前的位置 (使得 track.xywh 已经处于背景静止的坐标系下)
        new_xywh = self.warp_bboxes(self.xywh, warp_matrix)
        self._tlwh = self.xywh_to_tlwh(new_xywh)

        # 2. 批量扭曲 xywh_amemory (比如 30 帧历史)
        if len(self.xywh_amemory) > 0:
            amemory_arr = np.array(self.xywh_amemory)  # (M, 4)
            warped_amemory = self.warp_bboxes(amemory_arr, warp_matrix)
            self.xywh_amemory = deque(warped_amemory, maxlen=self.xywh_amemory.maxlen)

        # 3. 批量扭曲网络所需条件 conds (比如 10 帧历史)
        if len(self.conds) > 0:
            conds_arr = np.array(self.conds)  # (L, 8)
            hist_xywh = conds_arr[:, :4]  # (L, 4)
            delta = conds_arr[:, 4:]  # (L, 4)

            # 计算上一帧的绝对坐标
            hist_prev_xywh = hist_xywh - delta

            # 🚀 极致优化：将 hist_xywh 和 hist_prev_xywh 上下拼起来，一次性扔进 C++ 底层算完！
            combined_bboxes = np.vstack([hist_xywh, hist_prev_xywh])  # (2L, 4)
            warped_combined = self.warp_bboxes(combined_bboxes, warp_matrix)

            # 再把它们切分拆开
            new_hist_xywh = warped_combined[:len(conds_arr)]
            new_hist_prev_xywh = warped_combined[len(conds_arr):]

            # 重新计算向量差，并水平拼接为 (L, 8)
            new_delta = new_hist_xywh - new_hist_prev_xywh
            new_conds = np.concatenate([new_hist_xywh, new_delta], axis=1)

            self.conds = deque(new_conds, maxlen=self.conds.maxlen)

    # def warp_bbox(bbox_cxcywh, warp_matrix):
    #     """对 cx, cy, w, h 进行仿射变换"""
    #     cx, cy, w, h = bbox_cxcywh
    #     # 1. 变换中心点位置
    #     pt = np.array([[cx], [cy], [1.0]])
    #     new_pt = warp_matrix @ pt
    #     new_cx, new_cy = new_pt[0, 0], new_pt[1, 0]
    #
    #     # 2. 变换宽高 (根据仿射矩阵的尺度因子)
    #     # 严格的尺度计算公式 (处理了可能的旋转和缩放)
    #     scale_w = np.sqrt(warp_matrix[0, 0] ** 2 + warp_matrix[0, 1] ** 2)
    #     scale_h = np.sqrt(warp_matrix[1, 0] ** 2 + warp_matrix[1, 1] ** 2)
    #     new_w = w * scale_w
    #     new_h = h * scale_h
    #
    #     return np.array([new_cx, new_cy, new_w, new_h], dtype=np.float32)

    # def apply_camera_motion(self, warp_matrix):
    #     """
    #     🚀 核心：当新的一帧到来时，根据 GMC 矩阵把所有历史信息扭曲到当前相机的视角！
    #     """
    #     # 1. 扭曲当前的位置 (使得 track.xywh 已经处于背景静止的坐标系下)
    #     new_xywh = self.warp_bbox(self.xywh, warp_matrix)  # self.xywh由前一帧warp后的self._tlwh
    #     self._tlwh = self.xywh_to_tlwh(new_xywh)
    #
    #     # 🚀 修复 1：必须把历史平滑轨迹也扭曲到当前视角！否则速度计算会爆炸
    #     for i in range(len(self.xywh_amemory)):
    #         self.xywh_amemory[i] = self.warp_bbox(self.xywh_amemory[i], warp_matrix)  # 不断warp到当前帧视角
    #
    #     for i in range(len(self.conds)):
    #         cond = self.conds[i]
    #         hist_xywh = cond[:4]
    #         delta = cond[4:]
    #
    #         new_hist_xywh = self.warp_bbox(hist_xywh, warp_matrix)
    #         hist_prev_xywh = hist_xywh - delta
    #         new_hist_prev_xywh = self.warp_bbox(hist_prev_xywh, warp_matrix)
    #         new_delta = new_hist_xywh - new_hist_prev_xywh
    #
    #         self.conds[i] = np.concatenate([new_hist_xywh, new_delta])  # 替换为当前帧视角

    @staticmethod
    def multi_predict_prior(stracks, model, img_w, img_h, step=1):
        """批量计算网络先验，用于后续做巴氏距离匹配"""
        if len(stracks) == 0: return

        # 🚀 修复：同样动态获取 device
        device = next(model.parameters()).device
        batch_conds, batch_vel_masks = [], []
        # 🚀 修正 1：构建 8 维全局归一化向量
        img_scale_8d = np.array([img_w, img_h, img_w, img_h] * 2, dtype=np.float32)
        for st in stracks:
            # 🚀 严格复刻 Dataset 的归一化逻辑
            conds_array = np.array(st.conds)  # Shape: [L, 8]已经warp后的历史条件

            # 🚀 修正 2：动态 Padding！保证输入序列长度永远是 conds_len (10)
            pad_len = st.conds_len - len(conds_array)
            if pad_len > 0:
                pad_arr = np.zeros((pad_len, 8), dtype=np.float32)
                # 把不足的帧用 0 补在前面 (因为最新的帧在末尾)
                conds_array = np.concatenate([pad_arr, conds_array], axis=0)

            # 🚀 修正 3：去掉之前错误的 np.concatenate，直接全局归一化
            norm_cond = conds_array / img_scale_8d

            # 拼接并加入 Batch
            batch_conds.append(norm_cond)
            batch_vel_masks.append(st.vel_mask)

        # 构造 Batch 送入网络
        batch = {
            "conditions": torch.tensor(np.stack(batch_conds), dtype=torch.float32).to(device),
            "vel_mask": torch.tensor(np.stack(batch_vel_masks), dtype=torch.bool).to(device),
        }

        v_prior_batch, trans, cond_encoded = model.predict_prior(batch)  # B,4; B,d; B,1,d

        for i, st in enumerate(stracks):
            v_prior = v_prior_batch[i]

            # 把相对速度反归一化为像素尺度
            wh_4d = np.array([img_w, img_h, img_w, img_h])
            raw_vel = v_prior * wh_4d * step

            # 更新预测位置
            pred_xywh = st.xywh.copy() + raw_vel
            st.pred_xywh = pred_xywh

            # 🚀 必须更新当前坐标，保证哪怕没匹配上也能惯性滑行
            st._tlwh = STrack.xywh_to_tlwh(pred_xywh)

            st.xywh_pmemory.append(st.pred_xywh.copy())
            st.v_prior = v_prior.copy()  # 暂存给 opt 步骤用
            st.trans = trans[i].copy()
            st.cond_encoded = cond_encoded[i].copy()

    def apply_v_opt(self, v_opt, img_w, img_h, step=1):
        """计算出 v_opt 后的最终真值更新"""
        # 🚀 修复 3：必须基于上一帧的状态 (P_{t-1}) 加上最优速度，而不能用 self.xywh (P_prior)
        last_xywh = self.xywh_amemory[-1]
        # last_wh = last_xywh[2:4]
        # wh_4d = np.concatenate([last_wh, last_wh])
        wh_4d = np.array([img_w, img_h, img_w, img_h])
        raw_vel = v_opt * wh_4d * step

        final_xywh = last_xywh + raw_vel
        self._tlwh = STrack.xywh_to_tlwh(final_xywh)
        self.xywh_amemory.append(final_xywh)

    def update_mask(self):
        """提取更新 mask 的公共逻辑"""
        active_frames = self.tracklet_len + 1
        valid_count = min(active_frames, self.conds_len)
        new_mask = [False] * (self.conds_len - valid_count) + [True] * valid_count
        self.vel_mask.clear()
        self.vel_mask.extend(new_mask)

    def activate(self, frame_id):
        """Start a new tracklet"""
        """
        根据当前轨迹内的帧序号（从1开始）更新 vel_mask
        Args:
            frame_id_in_track: int, 当前是该 track 的第几帧（>=1）
        """
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id
        self.xywh_omemory.append(self.xywh.copy())
        self.xywh_pmemory.append(self.xywh.copy())
        self.xywh_amemory.append(self.xywh.copy())

        tmp_conds = np.concatenate((self.xywh.copy(), np.zeros_like(self.xywh)))
        self.conds.append(tmp_conds)

        # 🚀 修复 4：直接调用你写好的统一 update_mask，避免 valid_count 越界
        self.update_mask()

    def re_activate(self, new_track, frame_id, new_id=False, is_network_updated=True):
        step = frame_id - self.frame_id
        if step <= 0: step = 1  # 保护机制
        self.tracklet_len = 0  # 找回后重新计数生命
        self.frame_id = frame_id
        self.xywh_omemory.append(new_track.xywh.copy())
        # 🚀 只有当未使用网络平滑时，才采用检测框的原生坐标
        if not is_network_updated:
            self._tlwh = new_track.tlwh
            self.xywh_amemory.append(self.xywh.copy())

        if len(self.xywh_amemory) >= 2:
            tmp_delta_bbox = (self.xywh.copy() - self.xywh_amemory[-2].copy()) / step
        else:
            tmp_delta_bbox = np.zeros_like(self.xywh)

        tmp_conds = np.concatenate((self.xywh.copy(), tmp_delta_bbox))
        self.conds.append(tmp_conds)  # 🚀 必须是 append
        self.update_mask()

        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, is_network_updated=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :return:
        """
        # 🚀 修复 3：动态计算跨度 step
        step = frame_id - self.frame_id
        if step <= 0: step = 1

        self.frame_id = frame_id
        self.tracklet_len += 1

        self.xywh_omemory.append(new_track.xywh.copy())
        # self.xywh_amemory[-1] = self.xywh.copy()
        # 🚀 保护网络辛辛苦苦平滑出来的最优状态不被噪声覆盖
        if not is_network_updated:
            self._tlwh = new_track.tlwh
            self.xywh_amemory.append(self.xywh.copy())

        if len(self.xywh_amemory) >= 2:
            tmp_delta_bbox = (self.xywh.copy() - self.xywh_amemory[-2].copy()) / step
        else:
            tmp_delta_bbox = np.zeros_like(self.xywh)

        tmp_conds = np.concatenate((self.xywh.copy(), tmp_delta_bbox))
        self.conds.append(tmp_conds)  # 🚀 必须是 append
        self.update_mask()

        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @property
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
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] = ret[:2] + ret[2:] / 2
        # ret[2:] += ret[:2]
        return ret

    @staticmethod
    def xywh_to_tlwh(xywh):
        """新增的必备工具函数"""
        ret = np.asarray(xywh).copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
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
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class NeuralTBDtracker(object):
    def __init__(self, config, model, frame_rate=30):
        self.config = config
        self.model = model
        self.tracked_stracks = []  # type: list[STrack] # active tracks
        self.lost_stracks = []  # type: list[STrack]  # lost tracks
        self.removed_stracks = []  # type: list[STrack]  # removed tracks

        self.frame_id = 0
        self.seq_len = config.interval + 1  # 比如 10
        self.det_thresh = self.config.high_thres

        self.buffer_size = int(frame_rate / 30.0 * 30)
        self.max_time_lost = self.buffer_size

        self.gmc = GMC(method='sparseOptFlow', downscale=2)

        self.mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)

    def _apply_network_update(self, matches, stracks, detections, img_w, img_h, frame_id, tag):
        """
        🚀 封装核心的网络更新逻辑，供一阶段和二阶段复用,只有匹配上的轨迹才有资格更新速度
        """
        if len(matches) == 0:
            return []

        device = next(self.model.parameters()).device

        batch_v_inst, batch_v_prior, batch_trans, batch_cond_encoded = [], [], [], []
        matched_stracks = []
        steps = []  # 记录各自的时间跨度

        # 🚀 修正 5：定义全局尺度
        wh_4d = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)

        for itracked, idet in matches:
            track = stracks[itracked]
            det = detections[idet]

            # 🚀 修复 2：必须用检测框减去 P_{t-1}，而不能减去 track.xywh(P_prior)
            last_xywh = track.xywh_amemory[-1]
            step = frame_id - track.frame_id
            if step <= 0: step = 1
            # last_wh = last_xywh[2:4]
            # wh_4d = np.concatenate([last_wh, last_wh])
            raw_inst_vel = (det.xywh - last_xywh) / step  # step = 1
            v_inst = raw_inst_vel / wh_4d

            batch_v_prior.append(track.v_prior)
            batch_trans.append(track.trans)
            batch_cond_encoded.append(track.cond_encoded)
            batch_v_inst.append(v_inst)
            matched_stracks.append(track)
            steps.append(step)  # 🚀 必须确保它在 for 循环内部！

        batch_update = {
            "v_inst": torch.tensor(np.stack(batch_v_inst), dtype=torch.float32).to(device),
            "v_prior": torch.tensor(np.stack(batch_v_prior), dtype=torch.float32).to(device),
            "trans": torch.tensor(np.stack(batch_trans), dtype=torch.float32).to(device),
            "cond_encoded": torch.tensor(np.stack(batch_cond_encoded), dtype=torch.float32).to(device),
        }

        v_opt_batch, k_t_batch = self.model.predict_opt(batch_update)  # ,_

        for i, track in enumerate(matched_stracks):
            # 执行网络预测真值更新
            track.apply_v_opt(v_opt_batch[i], img_w, img_h, step=steps[i])
            # ==========================================
            # 提取对应轨迹的门控值并打印
            # ==========================================
            k_t = k_t_batch[i]  # 提取当前轨迹的 K_t [4维向量]

            # xy 代表平移运动的信任度，wh 代表尺度形变的信任度
            k_xy = (k_t[0] + k_t[1]) / 2.0
            k_wh = (k_t[2] + k_t[3]) / 2.0

            # 仅盯着你想要的 track_id (比如 24) 写入日志，避免文件过大
            if track.track_id == 22 and 'MultiUAV-007' in tag:
                # 打印到控制台
                print(f"[Gate] Frame: {frame_id:04d} | ID: {track.track_id:03d} | K_xy: {k_xy:.3f} | K_wh: {k_wh:.3f}")

                # 写入日志文件，方便后续画图 (以逗号分隔: frame, k_xy, k_wh)
                with open(f"gate_log_{tag.split(':')[0]}_id{track.track_id}.txt", "a") as f:
                    f.write(f"{frame_id},{k_xy:.4f},{k_wh:.4f}\n")

                # 2. 新增：记录速度日志
                # 处理 Tensor 与 Numpy 的转换
                v_opt_np = v_opt_batch[i].detach().cpu().numpy() if torch.is_tensor(v_opt_batch[i]) else np.array(
                    v_opt_batch[i])
                v_inst_np = batch_v_inst[i]

                # 反归一化为像素尺度 (Pixels / Frame)
                v_inst_pixel = v_inst_np * wh_4d * steps[i]
                v_opt_pixel = v_opt_np * wh_4d * steps[i]

                # 提取中心点 X 和 Y 方向的速度
                v_inst_x, v_inst_y = v_inst_pixel[0], v_inst_pixel[1]
                v_opt_x, v_opt_y = v_opt_pixel[0], v_opt_pixel[1]

                # 写入单独的速度日志文件
                with open(f"vel_log_{tag.split(':')[0]}_id{track.track_id}.txt", "a") as f:
                    f.write(f"{frame_id},{v_inst_x:.4f},{v_inst_y:.4f},{v_opt_x:.4f},{v_opt_y:.4f}\n")

        return matched_stracks

    def update(self, dets_load, frame_id, img_w, img_h, tag, img=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # ==========================================
        # 🚀 第一步：计算并应用 GMC (Camera Motion Compensation)
        # ==========================================
        if img is not None:
            # 获取 2x3 仿射变换矩阵
            # 传一个空的 dets 进去避免提取到目标身上的特征点
            empty_dets = np.zeros((0, 5))
            warp_matrix = self.gmc.apply(img, empty_dets)
        else:
            warp_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        # 🚀 在做任何预测和匹配前，先把所有现存轨迹拉到当前帧视角！
        for track in self.tracked_stracks + self.lost_stracks:  #  + self.removed_stracks
            if not track.is_activated and track.state != TrackState.Tracked and track.state != TrackState.Lost:
                continue
            track.apply_camera_motion(warp_matrix)
        # ==========================================

        dets = dets_load.copy()
        # dets[:, 2] = dets[:, 0] + dets[:, 2]
        # dets[:, 3] = dets[:, 1] + dets[:, 3]  # tlwh->tlbr
        remain_inds = dets[:, 4] > self.det_thresh
        inds_low = dets[:, 4] > self.config.low_thres
        inds_high = dets[:, 4] < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        dets = dets[remain_inds]

        '''Detections'''
        detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], None, self.config.interval, 30) for tlbrs in
                      dets[:, :5]] if len(dets) > 0 else []
        detections_second = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], None, self.config.interval, 30) for tlbrs
                             in dets_second[:, :5]] if len(dets_second) > 0 else []
        unconfirmed = [track for track in self.tracked_stracks if not track.is_activated]
        tracked_stracks = [track for track in self.tracked_stracks if track.is_activated]

        ''' Step 2: First association '''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # 1. 神经网络先验预测 (赋予 STrack.pred_tlwh)
        STrack.multi_predict_prior(strack_pool, self.model, img_w, img_h)
        # STrack.multi_predict_diff(strack_pool, self.model, img_w, img_h)

        dists = self.get_bhattacharyya_cost(strack_pool, detections)
        # dists = matching.iou_distance(strack_pool, detections)
        # iou_matrix = 1 - dists

        # 将距离转为代价矩阵，进行匈牙利匹配
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.9)  # 0.6

        # 🚀 提取为公用函数：执行网络融合并更新
        matched_stracks = self._apply_network_update(matches, strack_pool, detections, img_w, img_h, self.frame_id, tag)

        # 3. 状态机更新 (🚀 告诉 track 它的状态已经被网络修过了！)
        for i, track in enumerate(matched_stracks):
            det = detections[matches[i, 1]]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, is_network_updated=True)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, is_network_updated=True)
                refind_stracks.append(track)

        ''' Step 3: Second association, for low-score detections '''
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists_second = self.get_bhattacharyya_cost(r_tracked_stracks, detections_second)
        matches_second, u_track_second, u_detection_second = matching.linear_assignment(dists_second, thresh=0.8)

        # 🚀 第二阶段也要执行网络平滑！过滤掉低分框的巨大坐标抖动
        matched_stracks_second = self._apply_network_update(matches_second, r_tracked_stracks, detections_second, img_w, img_h, self.frame_id, tag)

        for i, track in enumerate(matched_stracks_second):
            det = detections_second[matches_second[i, 1]]  # 提取真实的低分检测框
            if track.state == TrackState.Tracked:
                # 缺失的 update 补上
                track.update(det, self.frame_id, is_network_updated=True)
                activated_starcks.append(track)
            else:
                # 修复自己传自己的 Bug，传入真正的 det
                track.re_activate(det, self.frame_id, new_id=False, is_network_updated=True)
                refind_stracks.append(track)

        # 把剩下的标记为 Lost
        for it in u_track_second:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # 未确认轨迹直接用 IoU 或 巴氏距离匹配，不做网络预测，因为缺乏历史特征 高置信度检测没有匹配上的大概率是新出现的目标
        rem_detections = [detections[i] for i in u_detection]
        dists_unconf = self.get_bhattacharyya_cost(unconfirmed, rem_detections)
        # dists = matching.iou_distance(unconfirmed, detections)
        matches_unconf, u_unconfirmed, u_detection_new = matching.linear_assignment(dists_unconf, thresh=0.7)
        # detections = [detections[i] for i in u_detection]
        # dists = self.get_bhattacharyya_cost(unconfirmed, detections)
        # matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches_unconf:
            det = rem_detections[idet]
            # 无网络预测，状态由检测框原样提供
            unconfirmed[itracked].update(det, self.frame_id, is_network_updated=False)
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection_new:
            track = rem_detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

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

        return output_stracks

    def get_bhattacharyya_cost(self, tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float64)  # 使用 float64

        tracks_xywh = np.array([t.xywh for t in tracks], dtype=np.float64)
        dets_xywh = np.array([d.xywh for d in detections], dtype=np.float64)

        track_vars = (tracks_xywh[:, 2:] / 2.0) ** 2 + 1e-6  #1e-6
        det_vars = (dets_xywh[:, 2:] / 2.0) ** 2 + 1e-6  #1e-6

        track_means = tracks_xywh[:, :2]
        det_means = dets_xywh[:, :2]
        sigma_avg = (track_vars[:, None, :] + det_vars[None, :, :]) / 2.0

        det_sigma_avg = sigma_avg[:, :, 0] * sigma_avg[:, :, 1]
        det_sigma_p = track_vars[:, 0] * track_vars[:, 1]
        det_sigma_q = det_vars[:, 0] * det_vars[:, 1]

        term2 = 0.5 * np.log(det_sigma_avg / (np.sqrt(det_sigma_p[:, None] * det_sigma_q[None, :]) + 1e-9))
        diff = track_means[:, None, :] - det_means[None, :, :]
        term1 = 0.125 * np.sum((diff ** 2) / sigma_avg, axis=2)
        bd_dist = term1 + term2

        # 归一化到 [0, 1] 供匈牙利算法使用 (这是一个非线性的映射，增强区分度)
        cost = 1.0 - np.exp(-bd_dist)

        return cost

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
    # pairs = np.where(pdist < 0.)
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
