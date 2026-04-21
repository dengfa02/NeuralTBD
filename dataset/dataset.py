import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tracker.gmc import GMC
# from Tracking.NeuralTBD.tracker.gmc import GMC
from tqdm import tqdm  # 用于显示离线计算的进度条
import cv2


class MultiUAVDataset(Dataset):
    def __init__(self, path, config, is_train=True):
        self.config = config
        self.is_train = is_train  # 控制训练/评估模式
        # seq_len 是网络需要的总帧数 (例如: 5帧历史 + 1帧当前 = 6)
        self.seq_len = self.config.interval + 1

        self.trackers = {}
        self.images = {}
        self.nframes = {}
        self.seq_dims = {}
        self.gmc_data = {}  # 🚀 新增：存储每个序列的 GMC 矩阵字典

        self.samples_index = []  # 存储所有合法样本的索引映射: (seq_name, track_idx, valid_start_idx)

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

                # ==========================================
                # 🚀 新增逻辑：检查并自动生成 GMC 文件
                # ==========================================
                gmc_file = os.path.join(self.gmc_dir, f"{seq}.txt")
                if not os.path.exists(gmc_file):
                    print(f"\n[GMC Cache] 找不到 {seq} 的运动补偿文件，开始自动计算并缓存...")
                    self._generate_and_save_gmc(seq, self.images[seq], gmc_file)

                # 读取生成好的 GMC 文件
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

                # 读取GT [frame, id, x, y, w, h, ...] (MOT格式默认x,y是左上角)
                gt_data = np.loadtxt(gt_file, dtype=np.float32, delimiter=',')
                unique_ids = np.unique(gt_data[:, 1])  # 寻找轨迹ID
                self.trackers[seq] = []

                track_idx = 0
                for track_id in unique_ids:
                    # 提取单条轨迹并按帧号排序
                    track = gt_data[gt_data[:, 1] == track_id]
                    track = track[track[:, 0].argsort()]

                    # 只有轨迹长度大于需要的序列长度，才有训练价值
                    if len(track) >= self.seq_len:
                        self.trackers[seq].append(track)  # 存储每条合格轨迹的完整数据
                        # 记录可供采样的起点索引
                        # 假设最大抽帧间隔为 max_step=3，为了保证不越界，合法起点会减少
                        max_step = self.config.get('max_step', 1)  # 最低为1
                        required_len = (self.seq_len - 1) * max_step + 1
                        if len(track) >= required_len:
                            for start_idx in range(len(track) - required_len + 1):
                                self.samples_index.append((seq, track_idx, start_idx))  # 计算训练中绝对安全的采样起点
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
        # 映射中心点
        new_cx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
        new_cy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
        # 映射宽高 (近似缩放)
        scale_x = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
        scale_y = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)
        new_w = w * scale_x
        new_h = h * scale_y
        return np.array([new_cx, new_cy, new_w, new_h])

    # ==========================================
    # 🚀 新增：离线 GMC 计算核心函数
    # ==========================================
    def _generate_and_save_gmc(self, seq, image_paths, save_path):
        """
        遍历视频的所有帧，调用 gmc.py 计算仿射变换矩阵，并保存为 txt 缓存。
        """
        # 实例化你的 GMC 计算器 (使用稀疏光流或 ORB)
        # 根据你传给我的 gmc.py，默认参数 downscale=2 可加速计算
        gmc_calculator = GMC(method='sparseOptFlow', downscale=2)

        results = []
        # 第一帧的仿射矩阵永远是单位阵 (因为没有上一帧做参考)
        results.append([1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        # 因为 GMC 是算相邻两帧，gmc.apply 内部会保存上一帧的状态，
        # 所以我们只需要顺序把图片喂进去即可。
        for i in tqdm(range(len(image_paths)), desc=f"Calculating GMC for {seq}"):
            frame_id = i + 1  # 对应 MOT 的 frame 索引
            img = cv2.imread(image_paths[i])
            if img is None:
                continue

            # 第 1 帧用于初始化 gmc 的内部状态 (prev_frame)，返回单位阵，跳过记录
            if i == 0:
                empty_dets = np.zeros((0, 5))
                _ = gmc_calculator.apply(img, empty_dets)
                continue

            # 对于第 2 帧及以后，传入图片和空的检测框
            # (由于无人机目标极小，不传入检测框做 mask 基本不影响背景光流点的提取)
            empty_dets = np.zeros((0, 5))
            warp_matrix = gmc_calculator.apply(img, empty_dets)

            # 把 2x3 的矩阵展平存入列表: [frame_id, a00, a01, a02, a10, a11, a12]
            row = [frame_id] + warp_matrix.flatten().tolist()
            results.append(row)

        # 将结果保存到硬盘，下次启动直接秒读！
        np.savetxt(save_path, np.array(results), fmt='%.8f', delimiter=',')
        print(f"[GMC Cache] 序列 {seq} 计算完毕，已缓存至: {save_path}")

    def __getitem__(self, idx):
        seq, track_idx, start_idx = self.samples_index[idx]
        track = self.trackers[seq][track_idx]
        img_w, img_h = self.seq_dims[seq]
        # 构造图像级尺度向量 [img_w, img_h, img_w, img_h]
        img_scale = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)

        # ==========================================
        # 1. 模拟 MOTIP: 动态时间间隔采样 (Time Dilation)
        # ==========================================
        max_step = self.config.get('max_step', 1)
        # 随机决定本次采样的帧间隔 (例如 1, 2, 或 3)
        step = random.randint(1, max_step) # random.randint(1, max_step)  固定3

        # 按step抽取 seq_len 个帧 (例如 6 帧)
        sampled_track = track[start_idx: start_idx + self.seq_len * step: step]

        # 提取帧号和 bbox
        frames = sampled_track[:, 0].astype(int)
        current_frame = frames[-1]  # 最后一帧是基准帧

        # ==========================================
        # 🚀 GMC 累积变换与历史框提取
        # ==========================================
        history_bboxes = []
        M_cum_first = None  # 【修复 1】：新增变量，专门用来保存第一帧到当前帧的矩阵
        for i in range(self.seq_len - 1):
            past_frame = frames[i]
            bbox = self.tlwh_to_cxcywh(sampled_track[i, 2:6])
            # 累乘计算从 past_frame 到 current_frame 的映射矩阵 M_cum
            M_cum = np.eye(3)
            # 举例: 要把 f=3 映射到 f=5，需要 M_4(将3映射到4) 和 M_5(将4映射到5)
            # 所以 p_5 = M_5 * M_4 * p_3
            for f in range(past_frame + 1, current_frame + 1):
                # 获取 f 帧的仿射矩阵 (如果没有则使用单位阵)
                M_f = self.gmc_data[seq].get(f, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
                M_f_3x3 = np.vstack([M_f, [0, 0, 1]])
                # 矩阵左乘累加
                M_cum = M_f_3x3 @ M_cum
                # 将历史框映射到当前帧视角！
            # 当 i=0 时，意味着这是序列的第一帧，我们把这个跨度最大的矩阵存下来
            if i == 0:
                M_cum_first = M_cum.copy()
            aligned_bbox = self.warp_bbox(bbox, M_cum[:2, :])
            history_bboxes.append(aligned_bbox)

        history_bboxes = np.array(history_bboxes)  # Shape: [5, 4]
        gt_bbox = self.tlwh_to_cxcywh(sampled_track[-1, 2:6])

        # ==========================================
        # 图像可视化与对齐验证
        # ==========================================
        # MOT 帧号从 1 开始，必须减 1 才能对应正确的图片索引！
        # cur_img_path = self.images[seq][current_frame - 1]
        # his_img_path = self.images[seq][frames[0] - 1]  # 第一帧到最后一帧的对齐效果
        #
        # img_last = cv2.imread(cur_img_path)
        # img_his = cv2.imread(his_img_path)
        #
        # if img_last is None or img_his is None:
        #     print(f"[WARN] Image not found: {cur_img_path} or {his_img_path}")
        # else:
        #     h, w = img_last.shape[:2]
        #
        #     # 【修复 3】：使用 M_cum_first 进行仿射变换，而不是已经被覆盖的 M_cum
        #     warped_img_his = cv2.warpAffine(img_his, M_cum_first[:2, :], (w, h))
        #
        #     blended_img = cv2.addWeighted(img_last, 0.5, warped_img_his, 0.5, 0)
        #
        #     img_last_vis = img_last.copy()
        #
        #     # (建议：你可以尝试在这里把 history_bboxes[0] 画在 warped_img_his 上，验证框是否也正确映射)
        #
        #     vis_result = np.concatenate([img_last_vis, warped_img_his, blended_img], axis=1)
        #     save_dir = "gmc_vis"
        #     os.makedirs(save_dir, exist_ok=True)
        #     save_path = os.path.join(save_dir, f"{seq}_frame{current_frame}_align.jpg")
        #     cv2.imwrite(save_path, vis_result)
        #     print(f"[GMC Vis] Saved alignment check: {save_path}")

        # ==========================================
        # 2. 对历史帧投毒 (Occlusion / IDSW)
        # ==========================================
        # 以 20% 的概率对历史帧施加遮挡(丢失)模拟 布尔类型更适合做 Mask: True代表有效，False代表被遮挡
        occ_mask = np.ones((self.seq_len - 1,), dtype=bool)
        obs_bbox = gt_bbox.copy()  # cxcywh

        # ==========================================
        # 2. 对历史帧投毒 (Occlusion + Jitter)
        # ==========================================
        occ_mask = np.ones((self.seq_len - 1,), dtype=bool)

        # 2.1 遮挡投毒 (Occlusion)
        if random.random() < self.config.get('prob_occ', 0.2):
            num_occ = random.randint(1, 3)
            valid_mask_range = range(1, len(history_bboxes) - 1)
            if len(valid_mask_range) >= num_occ:
                occ_indices = random.sample(valid_mask_range, num_occ)
                for oi in occ_indices:
                    history_bboxes[oi] = 0.0
                    occ_mask[oi] = False

        # 2.2 历史轨迹抖动投毒 (极度重要：打破网络的先验依赖)
        # 如果不给历史框加噪声，网络会过度信任 v_prior，导致门控失效
        hist_jitter_prob = self.config.get('prob_hist_jitter', 0.3)
        for i in range(len(history_bboxes)):
            if occ_mask[i] and random.random() < hist_jitter_prob:
                # 对历史框施加微小的高斯噪声 (不超过宽高的 5%)
                history_bboxes[i, 0] += np.random.normal(0, 0.05) * history_bboxes[i, 2]
                history_bboxes[i, 1] += np.random.normal(0, 0.05) * history_bboxes[i, 3]

        # ==========================================
        # 3. 对当前观测帧投毒 (IDSW / False Positive / Jitter)
        # ==========================================
        obs_bbox = gt_bbox.copy()  # cxcywh

        # 建议配置概率：prob_idsw=0.1, prob_jitter=0.3
        # 概率配置：极度破坏(0.1) -> 严重错配(0.1) -> 轻微抖动(0.3)
        prob_extreme = self.config.get('prob_extreme', 0.1)  # 🚀 新增：毁灭性破坏
        prob_idsw = self.config.get('prob_idsw', 0.1)
        prob_jitter = self.config.get('prob_jitter', 0.3)  # 降到0.3以内！

        # 🚀 3.1 毁灭性破坏 (模拟飞窗、极度遮挡、完全错配)
        if random.random() < prob_extreme:
            # (1) 制造巨大的位移 (跳出正常物理极限，偏移自身宽高的 2~10 倍)
            shift_x = random.choice([-1, 1]) * obs_bbox[2] * random.uniform(2.0, 10.0)
            shift_y = random.choice([-1, 1]) * obs_bbox[3] * random.uniform(2.0, 10.0)
            obs_bbox[0] += shift_x
            obs_bbox[1] += shift_y

            # (2) 制造极度的形变 (突然放大 3~10 倍，或者缩小到 10%~30%)
            if random.random() < 0.5:
                obs_bbox[2] *= random.uniform(2.0, 5.0)
                obs_bbox[3] *= random.uniform(2.0, 5.0)
            else:
                obs_bbox[2] *= random.uniform(0.1, 0.3)
                obs_bbox[3] *= random.uniform(0.1, 0.3)

        if random.random() < prob_idsw:
            # 【高级投毒：模拟 IDSW / 错误关联】
            # 现象：Tracker 匹配到了另一个轨迹（比如相交的无人机）
            # 特征：位移并不一定非常大，但"速度方向"和"宽高"往往发生了突变

            # 随机生成一个与当前运动方向有偏差的干扰向量
            speed_mag = np.sqrt(obs_bbox[2] ** 2 + obs_bbox[3] ** 2) * random.uniform(0.5, 2.0)
            angle = random.uniform(0, 2 * np.pi)

            # 强行扭曲观测框，使其偏离真实的 GT
            obs_bbox[0] += np.cos(angle) * speed_mag
            obs_bbox[1] += np.sin(angle) * speed_mag
            # IDSW 时，由于目标姿态或识别成了别的东西，宽高通常也会变
            obs_bbox[2] *= random.uniform(0.8, 1.2)
            obs_bbox[3] *= random.uniform(0.8, 1.2)

        elif random.random() < prob_jitter:
            # 【常规投毒：检测器边框抖动】
            # 使用 clip 限制最大抖动幅度，防止把小目标抖出屏幕外！
            max_jitter_x = obs_bbox[2] * 0.15
            max_jitter_y = obs_bbox[3] * 0.15

            dx = np.clip(np.random.normal(0, 0.05) * obs_bbox[2], -max_jitter_x, max_jitter_x)
            dy = np.clip(np.random.normal(0, 0.05) * obs_bbox[3], -max_jitter_y, max_jitter_y)

            obs_bbox[0] += dx
            obs_bbox[1] += dy
            obs_bbox[2] *= random.uniform(0.9, 1.1)
            obs_bbox[3] *= random.uniform(0.9, 1.1)

        # ==========================================
        # 3. 只有训练时才对当前观测帧投毒 (Jitter / False Positive)
        # ==========================================

        # # 遮挡投毒
        # if random.random() < self.config.get('prob_occ', 0.2):
        #     num_occ = random.randint(1, 3)  # 随机mask 1 到 3 帧
        #     valid_mask_range = range(1, len(history_bboxes) - 1)  # 强制保留第 0 帧（轨迹起点）和最后 1 帧（计算观测速度的锚点）不被遮挡
        #     if len(valid_mask_range) >= num_occ:
        #         occ_indices = random.sample(valid_mask_range, num_occ)
        #         for oi in occ_indices:
        #             history_bboxes[oi] = 0.0
        #             occ_mask[oi] = False
        # # 检测噪声投毒
        # fp_prob = self.config.get('prob_fp', 0.05)
        # jitter_prob = self.config.get('prob_jitter', 0.8)
        #
        # if random.random() < fp_prob:
        #     offset_x = random.choice([-1, 1]) * obs_bbox[2] * random.uniform(1.5, 3.0)
        #     offset_y = random.choice([-1, 1]) * obs_bbox[3] * random.uniform(1.5, 3.0)
        #     obs_bbox[0] += offset_x
        #     obs_bbox[1] += offset_y
        # elif random.random() < jitter_prob:
        #     obs_bbox[0] += np.random.normal(0, 0.05) * obs_bbox[2]
        #     obs_bbox[1] += np.random.normal(0, 0.05) * obs_bbox[3]
        #     obs_bbox[2] *= random.uniform(0.8, 1.2)
        #     obs_bbox[3] *= random.uniform(0.8, 1.2)

        # ==========================================
        # 4. 物理学特征归一化 (极其重要！)
        # ==========================================
        # 我们不能直接把绝对像素输入给网络，必须转成相对于目标尺寸的速度 (delta)
        # 注意：因为我们使用了抽帧(step)，实际时间跨度是 step，所以计算单帧速度要除以 step！
        history_velocities = np.zeros((self.seq_len - 2, 4), dtype=np.float32)
        vel_mask = np.zeros((self.seq_len - 2,), dtype=bool)
        for i in range(1, self.seq_len - 1):
            # 只有当前帧和上一帧都有效，速度才有效
            is_valid = occ_mask[i] and occ_mask[i - 1]
            vel_mask[i-1] = is_valid
            if is_valid:
                # wh = history_bboxes[i, 2:4]  # 速度 = (当前位置 - 上一位置) / step
                # 注意此时的 history_bboxes 已经是同一个绝对相机视角下的坐标了！
                raw_vel = (history_bboxes[i] - history_bboxes[i - 1]) / step
                # 归一化：除以目标当时的宽高，转化为尺度无关的特征，即相对于目标移动了自身几倍距离
                history_velocities[i-1] = raw_vel / img_scale # img_scale # np.concatenate([wh, wh])

        # 计算当前的观测瞬时速度 v_inst 和真实 GT 速度 v_gt
        # 因为我们在上面强保了 history_bboxes[-1] 不被遮挡，这里绝对安全！
        # last_wh = history_bboxes[-1, 2:4]
        # wh_4d = np.concatenate([last_wh, last_wh])
        # 计算当前的观测瞬时速度 v_inst (带噪声的)
        raw_inst_vel = (obs_bbox - history_bboxes[-1]) / step
        v_inst = raw_inst_vel / img_scale #  wh_4d
        # 计算真实的 GT 瞬时速度 v_gt (纯净的，用于计算 Loss)
        raw_gt_vel = (gt_bbox - history_bboxes[-1]) / step
        v_gt = raw_gt_vel / img_scale# wh_4d

        norm_history_bboxes = history_bboxes / img_scale
        norm_obs_bbox = obs_bbox / img_scale
        norm_gt_bbox = gt_bbox / img_scale
        conds = np.concatenate((np.array(norm_history_bboxes)[1:], np.array(history_velocities)), axis=1)
        return {
            'conditions': torch.tensor(conds, dtype=torch.float32),  # [9, 8] (如果输入10帧，则有9个历史框+速度)
            # 'history_vels': torch.tensor(history_velocities, dtype=torch.float32),  # [4, 4] (如果输入5帧，则有4个历史速度)
            'v_inst': torch.tensor(v_inst, dtype=torch.float32),  # [4]    带噪声的当前观测速度
            'v_gt': torch.tensor(v_gt, dtype=torch.float32),  # [4]    真实的GT速度（用于监督 L_task）

            # 'history_bboxes': torch.tensor(norm_history_bboxes[1:], dtype=torch.float32),  # 归一化历史坐标框去除第一个以和速度拼接
            'obs_bbox': torch.tensor(norm_obs_bbox, dtype=torch.float32),  # 当前帧带噪声观测（检测）框
            'gt_bbox': torch.tensor(norm_gt_bbox, dtype=torch.float32),  # 当前帧GT检测框

            # 返回了两个 Mask，方便网络做精准处理
            'occ_mask': torch.tensor(occ_mask[1:], dtype=torch.bool),  # [B,N] Bbox是否有效，去除第一个便于保持一致
            'vel_mask': torch.tensor(vel_mask, dtype=torch.bool),  # [B,N] 速度是否有效

            'step': torch.tensor(step, dtype=torch.float32),  # 把时间间隔传给网络可能有奇效
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
        # config = Config(dict_config)  # 转化为对象
    config = EasyDict(config)
    data_path = '/data/dcy/MultiUAV/img_train'
    a = MultiUAVDataset(data_path, config)
    b = a[1700200]
    print(b)
    pass
