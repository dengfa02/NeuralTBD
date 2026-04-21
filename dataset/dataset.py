import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tracker.gmc import GMC
from tqdm import tqdm
import cv2


class MultiUAVDataset(Dataset):
    def __init__(self, path, config, is_train=True):
        self.config = config
        self.is_train = is_train  # 控制训练/评估模式
        self.seq_len = self.config.interval + 1

        self.trackers = {}
        self.images = {}
        self.nframes = {}
        self.seq_dims = {}
        self.gmc_data = {}

        self.samples_index = []

        if os.path.isdir(path):
            if 'MOT' in path:
                self.seqs = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13",
                             "MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
            else:
                self.seqs = [s for s in os.listdir(path) if s.startswith('MultiUAV')]
            self.seqs.sort()

            # 🚀 设置 GMC 保存根目录
            self.gmc_dir = self.config.get('gmc_dir', '/data/dcy/MultiUAV/gmc_txt')
            os.makedirs(self.gmc_dir, exist_ok=True)

            for seq in self.seqs:
                imagePath = os.path.join(path, seq, "*.jpg")
                self.images[seq] = sorted(glob.glob(imagePath))
                self.nframes[seq] = len(self.images[seq])

                if len(self.images[seq]) > 0:
                    first_img = cv2.imread(self.images[seq][0])
                    h, w = first_img.shape[:2]
                    self.seq_dims[seq] = (w, h)
                else:
                    self.seq_dims[seq] = (1.0, 1.0)

                gmc_file = os.path.join(self.gmc_dir, f"{seq}.txt")
                if not os.path.exists(gmc_file):
                    print(f"\n[GMC Cache] 找不到 {seq} 的运动补偿文件，开始自动计算并缓存...")
                    self._generate_and_save_gmc(seq, self.images[seq], gmc_file)

                self.gmc_data[seq] = {}
                gmc_raw = np.loadtxt(gmc_file, dtype=np.float32, delimiter=',')
                if gmc_raw.ndim == 1:
                    gmc_raw = gmc_raw[np.newaxis, :]
                for row in gmc_raw:
                    frame_id = int(row[0])
                    matrix = row[1:7].reshape(2, 3)
                    self.gmc_data[seq][frame_id] = matrix

                # 接着处理真实的 GT 标签...
                gt_file = os.path.join('/data/dcy/MultiUAV/TrainLabels', f"{seq}.txt")
                if not os.path.exists(gt_file):
                    continue

                # 读取GT [frame, id, x, y, w, h, ...]
                gt_data = np.loadtxt(gt_file, dtype=np.float32, delimiter=',')
                unique_ids = np.unique(gt_data[:, 1])
                self.trackers[seq] = []

                track_idx = 0
                for track_id in unique_ids:
                    # 提取单条轨迹并按帧号排序
                    track = gt_data[gt_data[:, 1] == track_id]
                    track = track[track[:, 0].argsort()]

                    # 只有轨迹长度大于需要的序列长度，才有训练价值
                    if len(track) >= self.seq_len:
                        self.trackers[seq].append(track)
                        max_step = self.config.get('max_step', 1)  # 最低为1
                        required_len = (self.seq_len - 1) * max_step + 1
                        if len(track) >= required_len:
                            for start_idx in range(len(track) - required_len + 1):
                                self.samples_index.append((seq, track_idx, start_idx))
                        track_idx += 1

        print('=' * 80)
        print('dataset summary')
        print(f'Total Valid Tracklet Samples: {len(self.samples_index)}')
        print('=' * 80)

    def __len__(self):
        return len(self.samples_index)

    def tlwh_to_cxcywh(self, bbox):
        """将 [top_left_x, top_left_y, w, h] 转为中心点坐标 [cx, cy, w, h]"""
        return np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]])

    def warp_bbox(self, bbox, M):
        """用 2x3 仿射矩阵 M 映射 bbox (cx, cy, w, h)"""
        cx, cy, w, h = bbox
        new_cx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
        new_cy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
        scale_x = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
        scale_y = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)
        new_w = w * scale_x
        new_h = h * scale_y
        return np.array([new_cx, new_cy, new_w, new_h])

    # ==========================================
    # 🚀 新增：离线 GMC 计算核心函数
    # ==========================================
    def _generate_and_save_gmc(self, seq, image_paths, save_path):
        gmc_calculator = GMC(method='sparseOptFlow', downscale=2)
        results = []
        results.append([1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        for i in tqdm(range(len(image_paths)), desc=f"Calculating GMC for {seq}"):
            frame_id = i + 1
            img = cv2.imread(image_paths[i])
            if img is None:
                continue

            if i == 0:
                empty_dets = np.zeros((0, 5))
                _ = gmc_calculator.apply(img, empty_dets)
                continue

            empty_dets = np.zeros((0, 5))
            warp_matrix = gmc_calculator.apply(img, empty_dets)

            row = [frame_id] + warp_matrix.flatten().tolist()
            results.append(row)

        np.savetxt(save_path, np.array(results), fmt='%.8f', delimiter=',')
        print(f"[GMC Cache] 序列 {seq} 计算完毕，已缓存至: {save_path}")

    def __getitem__(self, idx):
        seq, track_idx, start_idx = self.samples_index[idx]
        track = self.trackers[seq][track_idx]
        img_w, img_h = self.seq_dims[seq]
        img_scale = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)

        max_step = self.config.get('max_step', 1)
        step = random.randint(1, max_step)
        sampled_track = track[start_idx: start_idx + self.seq_len * step: step]
        frames = sampled_track[:, 0].astype(int)
        current_frame = frames[-1]  # 最后一帧是基准帧

        history_bboxes = []
        M_cum_first = None
        for i in range(self.seq_len - 1):
            past_frame = frames[i]
            bbox = self.tlwh_to_cxcywh(sampled_track[i, 2:6])
            M_cum = np.eye(3)
            for f in range(past_frame + 1, current_frame + 1):
                M_f = self.gmc_data[seq].get(f, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
                M_f_3x3 = np.vstack([M_f, [0, 0, 1]])
                # 矩阵左乘累加
                M_cum = M_f_3x3 @ M_cum
            if i == 0:
                M_cum_first = M_cum.copy()
            aligned_bbox = self.warp_bbox(bbox, M_cum[:2, :])
            history_bboxes.append(aligned_bbox)

        history_bboxes = np.array(history_bboxes)  # Shape: [5, 4]
        gt_bbox = self.tlwh_to_cxcywh(sampled_track[-1, 2:6])

        occ_mask = np.ones((self.seq_len - 1,), dtype=bool)

        if random.random() < self.config.get('prob_occ', 0.2):
            num_occ = random.randint(1, 3)
            valid_mask_range = range(1, len(history_bboxes) - 1)
            if len(valid_mask_range) >= num_occ:
                occ_indices = random.sample(valid_mask_range, num_occ)
                for oi in occ_indices:
                    history_bboxes[oi] = 0.0
                    occ_mask[oi] = False

        hist_jitter_prob = self.config.get('prob_hist_jitter', 0.3)
        for i in range(len(history_bboxes)):
            if occ_mask[i] and random.random() < hist_jitter_prob:
                history_bboxes[i, 0] += np.random.normal(0, 0.05) * history_bboxes[i, 2]
                history_bboxes[i, 1] += np.random.normal(0, 0.05) * history_bboxes[i, 3]

        obs_bbox = gt_bbox.copy()  # cxcywh

        prob_extreme = self.config.get('prob_extreme', 0.1)
        prob_idsw = self.config.get('prob_idsw', 0.1)
        prob_jitter = self.config.get('prob_jitter', 0.3)

        if random.random() < prob_extreme:
            shift_x = random.choice([-1, 1]) * obs_bbox[2] * random.uniform(2.0, 10.0)
            shift_y = random.choice([-1, 1]) * obs_bbox[3] * random.uniform(2.0, 10.0)
            obs_bbox[0] += shift_x
            obs_bbox[1] += shift_y

            if random.random() < 0.5:
                obs_bbox[2] *= random.uniform(2.0, 5.0)
                obs_bbox[3] *= random.uniform(2.0, 5.0)
            else:
                obs_bbox[2] *= random.uniform(0.1, 0.3)
                obs_bbox[3] *= random.uniform(0.1, 0.3)

        if random.random() < prob_idsw:
            speed_mag = np.sqrt(obs_bbox[2] ** 2 + obs_bbox[3] ** 2) * random.uniform(0.5, 2.0)
            angle = random.uniform(0, 2 * np.pi)

            obs_bbox[0] += np.cos(angle) * speed_mag
            obs_bbox[1] += np.sin(angle) * speed_mag
            obs_bbox[2] *= random.uniform(0.8, 1.2)
            obs_bbox[3] *= random.uniform(0.8, 1.2)

        elif random.random() < prob_jitter:
            max_jitter_x = obs_bbox[2] * 0.15
            max_jitter_y = obs_bbox[3] * 0.15

            dx = np.clip(np.random.normal(0, 0.05) * obs_bbox[2], -max_jitter_x, max_jitter_x)
            dy = np.clip(np.random.normal(0, 0.05) * obs_bbox[3], -max_jitter_y, max_jitter_y)

            obs_bbox[0] += dx
            obs_bbox[1] += dy
            obs_bbox[2] *= random.uniform(0.9, 1.1)
            obs_bbox[3] *= random.uniform(0.9, 1.1)

        history_velocities = np.zeros((self.seq_len - 2, 4), dtype=np.float32)
        vel_mask = np.zeros((self.seq_len - 2,), dtype=bool)
        for i in range(1, self.seq_len - 1):
            is_valid = occ_mask[i] and occ_mask[i - 1]
            vel_mask[i-1] = is_valid
            if is_valid:
                raw_vel = (history_bboxes[i] - history_bboxes[i - 1]) / step
                history_velocities[i-1] = raw_vel / img_scale

        raw_inst_vel = (obs_bbox - history_bboxes[-1]) / step
        v_inst = raw_inst_vel / img_scale
        raw_gt_vel = (gt_bbox - history_bboxes[-1]) / step
        v_gt = raw_gt_vel / img_scale

        norm_history_bboxes = history_bboxes / img_scale
        norm_obs_bbox = obs_bbox / img_scale
        norm_gt_bbox = gt_bbox / img_scale
        conds = np.concatenate((np.array(norm_history_bboxes)[1:], np.array(history_velocities)), axis=1)
        return {
            'conditions': torch.tensor(conds, dtype=torch.float32),  # [9, 8] (如果输入10帧，则有9个历史框+速度)
            'v_inst': torch.tensor(v_inst, dtype=torch.float32),  # [4]    带噪声的当前观测速度
            'v_gt': torch.tensor(v_gt, dtype=torch.float32),  # [4]    真实的GT速度（用于监督 L_task）
            'obs_bbox': torch.tensor(norm_obs_bbox, dtype=torch.float32),  # 当前帧带噪声观测（检测）框
            'gt_bbox': torch.tensor(norm_gt_bbox, dtype=torch.float32),  # 当前帧GT检测框
            'occ_mask': torch.tensor(occ_mask[1:], dtype=torch.bool),  # [B,N] Bbox是否有效，去除第一个便于保持一致
            'vel_mask': torch.tensor(vel_mask, dtype=torch.bool),  # [B,N] 速度是否有效

            'step': torch.tensor(step, dtype=torch.float32),
            'img_scale': torch.tensor(img_scale)  # 把图像尺度也传出去，方便后续推理时还原坐标
        }


if __name__ == "__main__":
    import yaml
    from easydict import EasyDict
    # 定义一个简单的包装类
    class Config:
        def __init__(self, entries):
            for k, v in entries.items():
                if isinstance(v, dict):
                    self.__dict__[k] = Config(v)  # 递归处理子字典
                else:
                    self.__dict__[k] = v
    config_path = '/home/dcy/small_target_detection/Tracking/NeuralTBD/configs/multiuav.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    data_path = '/data/dcy/MultiUAV/img_train'
    a = MultiUAVDataset(data_path, config)
    b = a[1700200]
    print(b)
    pass
