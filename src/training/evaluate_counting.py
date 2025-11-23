import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torchvision.ops as tvops
import json


# === Define bins ===
BINS = [
    (0, 2), (3, 5), (6, 20), (21, 30), (31, 40),
    (41, 50), (51, 100), (101, 150), (151, 200),
    (251, 300), ("300+", None)
]


def smape(y_true, y_pred, eps=1e-6):
    """Symmetric mean absolute percentage error (0–200%)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(
        np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps) * 2.0
    )


def soft_nms_gaussian(boxes, scores, sigma=0.5, iou_thresh=0.5, score_thresh=0.001):
    """
    Minimal Gaussian soft-NMS (CPU/Numpy).
    boxes: (N,4) xyxy
    scores: (N,)
    """
    if len(boxes) == 0:
        return []

    boxes = boxes.copy()
    scores = scores.copy()
    N = boxes.shape[0]
    idxs = np.arange(N)
    keep = []

    for i in range(N):
        maxpos = i + np.argmax(scores[i:])
        boxes[[i, maxpos]] = boxes[[maxpos, i]]
        scores[[i, maxpos]] = scores[[maxpos, i]]
        idxs[[i, maxpos]] = idxs[[maxpos, i]]

        if scores[i] < score_thresh:
            continue
        keep.append(int(idxs[i]))

        if i == N - 1:
            break

        # IoU with remaining
        xx1 = np.maximum(boxes[i, 0], boxes[i+1:, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[i+1:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[i+1:, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[i+1:, 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2]-boxes[i, 0]) * (boxes[i, 3]-boxes[i, 1])
        area_j = (boxes[i+1:, 2]-boxes[i+1:, 0]) * (boxes[i+1:, 3]-boxes[i+1:, 1])
        union = area_i + area_j - inter
        iou = np.zeros_like(inter)
        mask = union > 0
        iou[mask] = inter[mask] / union[mask]

        # Gaussian decay
        scores[i+1:] *= np.exp(-(iou ** 2) / sigma)

    return keep


@torch.no_grad()
def evaluate_counting(
    model,
    data_loader,
    device,
    max_batches=None,
    save_dir=None,
    epoch=None,
    score_grid=(0.3, 0.4, 0.5, 0.6),
    iou_grid=(0.3, 0.5),
    use_soft_nms=False,
    soft_sigma=0.5
):
    """
    Evaluate colony counting performance of Faster R-CNN model.

    Reconstructs plate coordinates from patch predictions, applies plate-level (soft-)NMS,
    and grid-searches {score_thresh, nms_iou} to minimize plate-level MAE.

    Returns:
      patch_metrics, plate_metrics, bin_metrics
    """

    model.eval()
    true_counts_patch, pred_counts_patch = [], []

    plate_true = {}
    plate_boxes = {}
    plate_scores = {}

    # --- Collect predictions and GT ---
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            # Patch-level
            true_count = len(target["boxes"])
            scores = output["scores"].cpu().numpy()
            pred_count = np.sum(scores > 0.5)  # fixed 0.5 for patch stats
            true_counts_patch.append(true_count)
            pred_counts_patch.append(pred_count)

            # Plate-level
            plate_id = int(target["image_id"].item())
            plate_true[plate_id] = plate_true.get(plate_id, 0) + true_count

            off_x, off_y = target["patch_origin"]
            H, W = target["orig_size"]

            boxes = output["boxes"].cpu().numpy()
            if boxes.size > 0:
                boxes[:, [0, 2]] += off_x
                boxes[:, [1, 3]] += off_y
                boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, W-1)
                boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, H-1)

            plate_boxes.setdefault(plate_id, []).append(boxes)
            plate_scores.setdefault(plate_id, []).append(scores)

        if max_batches is not None and batch_idx >= max_batches:
            break

    # Flatten per-plate
    plate_ids = sorted(plate_true.keys())
    for pid in plate_ids:
        if plate_boxes[pid]:
            plate_boxes[pid] = np.vstack(plate_boxes[pid])
            plate_scores[pid] = np.concatenate(plate_scores[pid])
        else:
            plate_boxes[pid] = np.zeros((0,4))
            plate_scores[pid] = np.zeros((0,))

    # --- Grid search ---
    best_mae = float("inf")
    best_cfg = (None, None)
    best_counts = None

    for sc in score_grid:
        for iou in iou_grid:
            preds = []
            for pid in plate_ids:
                boxes = plate_boxes[pid]
                scores = plate_scores[pid]
                if boxes.shape[0] == 0:
                    preds.append(0)
                    continue

                mask = scores >= sc
                boxes_thr = boxes[mask]
                scores_thr = scores[mask]
                if boxes_thr.shape[0] == 0:
                    preds.append(0)
                    continue

                if not use_soft_nms:
                    keep = tvops.nms(
                        torch.as_tensor(boxes_thr, dtype=torch.float32),
                        torch.as_tensor(scores_thr, dtype=torch.float32),
                        iou
                    ).cpu().numpy()
                else:
                    keep = soft_nms_gaussian(boxes_thr, scores_thr, sigma=soft_sigma,
                                             iou_thresh=iou, score_thresh=sc)
                preds.append(len(keep))

            true_plate_counts = [plate_true[i] for i in plate_ids]
            mae = mean_absolute_error(true_plate_counts, preds)
            if mae < best_mae:
                best_mae = mae
                best_cfg = (sc, iou)
                best_counts = preds

    # --- Save best thresholds to JSON ---
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)  
        best_json = save_dir / "best_postproc.json"
        best_info = {
            "epoch": epoch,
            "best_score_thresh": float(best_cfg[0]),
            "best_nms_iou": float(best_cfg[1]),
            "mae_plate": float(best_mae)
        }
        with open(best_json, "w") as f:
            json.dump(best_info, f, indent=4)

    # --- Final plate metrics ---
    pred_plate_counts = best_counts
    true_plate_counts = [plate_true[i] for i in plate_ids]

    mae_patch = mean_absolute_error(true_counts_patch, pred_counts_patch)
    rmse_patch = np.sqrt(mean_squared_error(true_counts_patch, pred_counts_patch))
    bias_patch = np.mean(np.array(pred_counts_patch) - np.array(true_counts_patch))
    msle_patch = mean_squared_log_error(true_counts_patch, pred_counts_patch)
    smape_patch = smape(true_counts_patch, pred_counts_patch)

    mae_plate = mean_absolute_error(true_plate_counts, pred_plate_counts)
    rmse_plate = np.sqrt(mean_squared_error(true_plate_counts, pred_plate_counts))
    bias_plate = np.mean(np.array(pred_plate_counts) - np.array(true_plate_counts))
    msle_plate = mean_squared_log_error(true_plate_counts, pred_plate_counts)
    smape_plate = smape(true_plate_counts, pred_plate_counts)

    # --- Bin metrics ---
    bin_metrics = {}
    for b in BINS:
        if isinstance(b[0], str):
            mask = [tc >= 300 for tc in true_plate_counts]
            label = "300+"
        else:
            low, high = b
            mask = [(tc >= low and tc <= high) for tc in true_plate_counts]
            label = f"{low}-{high}"
        if any(mask):
            tcs = np.array(true_plate_counts)[mask]
            pcs = np.array(pred_plate_counts)[mask]
            bin_metrics[label] = {
                "n": len(tcs),
                "mae": mean_absolute_error(tcs, pcs),
                "rmse": np.sqrt(mean_squared_error(tcs, pcs)),
                "bias": np.mean(pcs - tcs),
                "msle": mean_squared_log_error(tcs, pcs),
                "smape": smape(tcs, pcs),
            }
        else:
            bin_metrics[label] = {"n": 0, "mae": None, "rmse": None, "bias": None, "msle": None, "smape": None}

 
    print(f"\nEvaluation on {len(true_counts_patch)} patches from {len(plate_ids)} plates:")
    print("Patch-level metrics:")
    print(f"  MAE   = {mae_patch:.3f}")
    print(f"  RMSE  = {rmse_patch:.3f}")
    print(f"  Bias  = {bias_patch:.3f}")
    print(f"  MSLE  = {msle_patch:.5f}")
    print(f"  sMAPE = {smape_patch:.2%}")

    print("Plate-level metrics (after NMS/grid search):")
    print(f"  MAE   = {mae_plate:.3f}")
    print(f"  RMSE  = {rmse_plate:.3f}")
    print(f"  Bias  = {bias_plate:.3f}")
    print(f"  MSLE  = {msle_plate:.5f}")
    print(f"  sMAPE = {smape_plate:.2%}")
    print(f"  Best config: score_thresh={best_cfg[0]}, nms_iou={best_cfg[1]}")

    print("Binned plate-level metrics:")
    for label, m in bin_metrics.items():
        if m["n"] > 0:
            print(f"  {label} (n={m['n']}): "
                  f"MAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}, Bias={m['bias']:.3f}, "
                  f"MSLE={m['msle']:.5f}, sMAPE={m['smape']:.2%}")
        else:
            print(f"  {label} (n=0): no samples")

    # --- Save logs & plots ---
    if save_dir is not None:
        log_file = save_dir / "metrics.csv"
        row = {
            "epoch": epoch,
            "mae_patch": mae_patch,
            "rmse_patch": rmse_patch,
            "bias_patch": bias_patch,
            "msle_patch": msle_patch,
            "smape_patch": smape_patch,
            "mae_plate": mae_plate,
            "rmse_plate": rmse_plate,
            "bias_plate": bias_plate,
            "msle_plate": msle_plate,
            "smape_plate": smape_plate,
            "best_score_thresh": best_cfg[0],
            "best_nms_iou": best_cfg[1],
        }
        for label, m in bin_metrics.items():
            row[f"mae_bin_{label}"] = m["mae"]
            row[f"rmse_bin_{label}"] = m["rmse"]
            row[f"bias_bin_{label}"] = m["bias"]
            row[f"msle_bin_{label}"] = m["msle"]
            row[f"smape_bin_{label}"] = m["smape"]

        if log_file.exists():
            df = pd.read_csv(log_file)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(log_file, index=False)

        # Scatter plots
        plt.figure()
        plt.scatter(true_counts_patch, pred_counts_patch, alpha=0.5, s=8)
        if len(true_counts_patch) > 0:
            a, b = min(true_counts_patch), max(true_counts_patch)
            plt.plot([a, b], [a, b], "r--")
        plt.xlabel("True Count (patch)")
        plt.ylabel("Predicted Count (patch)")
        plt.title(f"Patch Scatter (epoch {epoch})")
        plt.savefig(save_dir / f"scatter_patch_epoch_{epoch}.png")
        plt.close()

        plt.figure()
        plt.scatter(true_plate_counts, pred_plate_counts, alpha=0.6, s=16)
        if len(true_plate_counts) > 0:
            a, b = min(true_plate_counts), max(true_plate_counts)
            plt.plot([a, b], [a, b], "r--")
        plt.xlabel("True Count (plate)")
        plt.ylabel("Predicted Count (plate)")
        plt.title(f"Plate Scatter (epoch {epoch})")
        plt.savefig(save_dir / f"scatter_plate_epoch_{epoch}.png")
        plt.close()

    return (
        (mae_patch, rmse_patch, bias_patch, msle_patch, smape_patch),
        (mae_plate, rmse_plate, bias_plate, msle_plate, smape_plate),
        bin_metrics
    )

