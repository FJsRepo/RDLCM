import numpy as np

def eval(detected_targets, gt_targets):
    detected_targets = np.array(detected_targets)
    gt_targets = np.array(gt_targets)
    target_detected_diff = []
    TD = 0  # 正确检测到的
    FD = 0  # 检测总数-其中的正确检测数量
    TU = 0  # 总正确数 - 被检测到的
    detected_list = []
    detectedTargetsNum = len(detected_targets)
    gtTargetsNum = len(gt_targets)
    for i in range(detectedTargetsNum):
        detected_target = detected_targets[i]
        for j in range(gtTargetsNum):
            if j not in detected_list:
                gt_target = gt_targets[j]
                distance = np.linalg.norm(detected_target - gt_target)
                if distance <= 5:
                    TD = TD + 1
                    detected_list.append(j)
                    target_detected_diff.append(distance)
    FD = len(detected_targets) - TD
    TU = len(gt_targets) - len(detected_list)
    td_diff = np.sum(target_detected_diff)

    return TD, FD, TU, td_diff