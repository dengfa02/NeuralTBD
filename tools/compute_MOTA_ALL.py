import os
import numpy as np
import motmetrics as mm
import trackeval

if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a: np.asarray(a, dtype=float)


def compute_iou(box1, box2):
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)), dtype=float)

    x1 = np.maximum(box1[:, 0][:, np.newaxis], box2[:, 0])
    y1 = np.maximum(box1[:, 1][:, np.newaxis], box2[:, 1])
    x2 = np.minimum(box1[:, 2][:, np.newaxis], box2[:, 2])
    y2 = np.minimum(box1[:, 3][:, np.newaxis], box2[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = area1[:, np.newaxis] + area2 - inter_area

    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou


def compute_hota_for_sequence(gt_file, res_file):
    gt_raw = mm.io.loadtxt(gt_file, fmt="mot16", min_confidence=1)
    ts_raw = mm.io.loadtxt(res_file, fmt="mot16")

    all_frames = sorted(list(set(gt_raw.index.get_level_values(0)) | set(ts_raw.index.get_level_values(0))))

    # --- 关键修复：收集所有唯一的 ID，并创建连续映射字典 ---
    unique_gt_ids = sorted(list(set(gt_raw.index.get_level_values(1)))) if not gt_raw.empty else []
    unique_trk_ids = sorted(list(set(ts_raw.index.get_level_values(1)))) if not ts_raw.empty else []

    gt_id_map = {orig_id: i for i, orig_id in enumerate(unique_gt_ids)}
    trk_id_map = {orig_id: i for i, orig_id in enumerate(unique_trk_ids)}
    # -------------------------------------------------------------

    data = {
        'num_timesteps': len(all_frames),
        'gt_ids': [],
        'tracker_ids': [],
        'similarity_scores': [],
        'num_gt_dets': 0,
        'num_tracker_dets': 0,
        'num_gt_ids': len(unique_gt_ids),
        'num_tracker_ids': len(unique_trk_ids)
    }

    for f in all_frames:
        # 1. 提取真值 (GT)
        if f in gt_raw.index:
            d_gt = gt_raw.loc[f]
            boxes_gt = np.atleast_2d(d_gt.iloc[:, :4].values).astype(float).copy()
            boxes_gt[:, 2:] += boxes_gt[:, :2]

            raw_ids_gt = np.atleast_1d(d_gt.index.values).astype(int)
            ids_gt = np.array([gt_id_map[x] for x in raw_ids_gt], dtype=int)
        else:
            boxes_gt = np.empty((0, 4), dtype=float)
            ids_gt = np.empty((0,), dtype=int)

        if f in ts_raw.index:
            d_ts = ts_raw.loc[f]
            boxes_ts = np.atleast_2d(d_ts.iloc[:, :4].values).astype(float).copy()
            boxes_ts[:, 2:] += boxes_ts[:, :2]

            raw_ids_ts = np.atleast_1d(d_ts.index.values).astype(int)
            ids_ts = np.array([trk_id_map[x] for x in raw_ids_ts], dtype=int)
        else:
            boxes_ts = np.empty((0, 4), dtype=float)
            ids_ts = np.empty((0,), dtype=int)

        iou_matrix = compute_iou(boxes_gt, boxes_ts)

        data['gt_ids'].append(ids_gt)
        data['tracker_ids'].append(ids_ts)
        data['similarity_scores'].append(iou_matrix)

        data['num_gt_dets'] += len(ids_gt)
        data['num_tracker_dets'] += len(ids_ts)

    hota_metric = trackeval.metrics.HOTA()
    return hota_metric.eval_sequence(data)


def process_directory(directory1, directory2, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "mota_hota_combined.txt")

    files = sorted([f for f in os.listdir(directory1) if f.endswith(".txt")])
    accs, names, hota_results = [], [], []

    print(f"Starting evaluation on {len(files)} sequences...\n")

    for i, file in enumerate(files):
        gt_f = os.path.join(directory1, file)
        res_f = os.path.join(directory2, file)
        if not os.path.exists(res_f):
            continue

        try:
            # --- MOTA 计算 ---
            gt = mm.io.loadtxt(gt_f, fmt="mot16", min_confidence=1)
            ts = mm.io.loadtxt(res_f, fmt="mot16")
            acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)

            # --- HOTA 计算 ---
            res_hota = compute_hota_for_sequence(gt_f, res_f)

            accs.append(acc)
            names.append(file)
            hota_results.append(res_hota)
            print(f"[{i + 1}/{len(files)}] Successfully processed: {file}")

        except Exception as e:
            print(f"[{i + 1}/{len(files)}] Error processing {file}: {e}")

    mh = mm.metrics.create()
    metrics_list = ['mota', 'idf1', 'num_false_positives', 'num_misses', 'num_switches', 'num_objects']
    summary_mm = mh.compute_many(accs, metrics=metrics_list, names=names, generate_overall=True)

    header = f"{'Sequence':<25} | {'MOTA':>8} | {'FP':>8} | {'FN':>8} | {'IDSW':>8} | {'IDF1':>8} | {'HOTA':>8} | {'DetA':>8} | {'AssA':>8} | {'GT':>8}"
    table_width = len(header)
    print("\n" + "=" * table_width + "\n" + header + "\n" + "-" * table_width)

    with open(output_file, 'w') as f:
        f.write(header + "\n" + "-" * table_width + "\n")

        for i, name in enumerate(names):
            # 获取各指标数据
            row_mm = summary_mm.loc[name]
            m = row_mm['mota'] * 100
            fp = int(row_mm['num_false_positives'])
            fn = int(row_mm['num_misses'])
            idsw = int(row_mm['num_switches'])
            id_score = row_mm['idf1'] * 100
            gt_count = int(row_mm['num_objects'])

            # HOTA 结果取多阈值平均
            h = np.mean(hota_results[i]['HOTA']) * 100
            d = np.mean(hota_results[i]['DetA']) * 100
            a = np.mean(hota_results[i]['AssA']) * 100

            line = f"{name:<25} | {m:>8.4f} | {fp:>8d} | {fn:>8d} | {idsw:>8d} | {id_score:>8.4f} | {h:>8.4f} | {d:>8.4f} | {a:>8.4f} | {gt_count:>8d}"
            print(line)
            f.write(line + "\n")

        overall_mm = summary_mm.loc['OVERALL']
        o_m = overall_mm['mota'] * 100
        o_fp = int(overall_mm['num_false_positives'])
        o_fn = int(overall_mm['num_misses'])
        o_idsw = int(overall_mm['num_switches'])
        o_id = overall_mm['idf1'] * 100
        o_gt = int(overall_mm['num_objects'])

        o_h = np.mean([np.mean(x['HOTA']) for x in hota_results]) * 100
        o_d = np.mean([np.mean(x['DetA']) for x in hota_results]) * 100
        o_a = np.mean([np.mean(x['AssA']) for x in hota_results]) * 100

        final_line = f"{'OVERALL':<25} | {o_m:>8.4f} | {o_fp:>8d} | {o_fn:>8d} | {o_idsw:>8d} | {o_id:>8.4f} | {o_h:>8.4f} | {o_d:>8.4f} | {o_a:>8.4f} | {o_gt:>8d}"
        print("-" * table_width + "\n" + final_line + "\n" + "=" * table_width)
        f.write("-" * table_width + "\n" + final_line + "\n")

        return o_m, o_id

if __name__ == '__main__':
    directory1 = r'/data/dcy/MultiUAV/ValLabels'
    directory2 = r'/home/dcy/small_target_detection/Tracking/NeuralTBD/experiments/output/predict/results'
    output_folder = r'/home/dcy/small_target_detection/Tracking/NeuralTBD/experiments/output/predict/combined_metrics'
    process_directory(directory1, directory2, output_folder)
