import json
from time import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.eval import eval

red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)

def draw_detected_image(image, imgName, targetPos, color):
    plt.close('all')
    plt.figure()
    targetNum = len(targetPos)
    for i in range(targetNum):
        plt.plot(targetPos[i][0]-1, targetPos[i][1]+5, marker='^', color=color)
    # plt.title(imgName)
    plt.axis('off')
    plt.imshow(image, 'gray')
    plt.savefig('final_detected_figures/'+imgName, bbox_inches='tight', pad_inches=0.0, dpi=300)
    # plt.show()

def draw_candidate_image(originalImg, imgName, candidatePos, candidateLen, SSLy_list):
    candidateNum = len(candidatePos)
    candidatePosCouple = []
    for i in range(candidateNum):
        candidatePosi = candidatePos[i]
        candidateLeni = candidateLen[i]
        targetFlag = True
        if targetFlag == True:
            if candidateLeni <= 2:
                candidatePosi = candidatePosi + 1
            elif candidateLeni <= 4:
                candidatePosi = candidatePosi + 2
            else:
                candidatePosi = candidatePosi + int(np.ceil(candidateLeni/2))

            candidatePosCouple.append([candidatePosi, SSLy_list[candidatePosi]])

    draw_detected_image(originalImg, imgName, candidatePosCouple, 'red')

def targets_anno_file(targets_anno_file_name):
    TargetImgNameList = []
    targets_anno = []
    with open(targets_anno_file_name, 'r') as anno_obj:
        lines = anno_obj.readlines()
    for line in lines:
        data = json.loads(line)
        raw_file = ((data['raw_file']).split('/'))[-1]
        targets_anno.append({
            'raw_file': raw_file,
            'point_targets': data['point_targets'],
            'rectangle_targets': data['rectangle_targets']
        })
        TargetImgNameList.append(raw_file)
    return TargetImgNameList, targets_anno

def SSLy(pred_left, pred_right):
    SSLy_list = []
    for x in range(384):
        x1 = pred_left[0]
        y1 = pred_left[1]
        x2 = pred_right[0]
        y2 = pred_right[1]
        k = (y2 - y1) / (x2 - x1)
        b = y1
        y = round(k * x + b)
        SSLy_list.append(y)

    return SSLy_list

def LCV(img, cellWidth, centralCPx, centralCPy):
    halfWidth = int(np.floor(cellWidth/2))
    patch0 = np.zeros((cellWidth, cellWidth))  # (x,y)
    patch1 = np.zeros((cellWidth, cellWidth))  # (x-cellWidth,y)
    patch2 = np.zeros((cellWidth, cellWidth))  # (x-cellWidth,y-cellWidth)
    patch3 = np.zeros((cellWidth, cellWidth))  # (x,y-cellWidth)
    patch4 = np.zeros((cellWidth, cellWidth))  # (x+cellWidth,y-cellWidth)
    patch5 = np.zeros((cellWidth, cellWidth))  # (x+cellWidth,y)
    patch6 = np.zeros((cellWidth, cellWidth))  # (x+cellWidth,y+cellWidth)
    patch7 = np.zeros((cellWidth, cellWidth))  # (x,y+cellWidth)
    patch8 = np.zeros((cellWidth, cellWidth))  # (x-cellWidth,y+cellWidth)
    # 3 -1 0 1
    # 5 -2 -1 0 1 2
    # (-np.floor(cellWidth/2), np.floor(cellWidth/2) + 1)
    for i in range(-halfWidth, halfWidth + 1):  # col
        for j in range(-halfWidth, halfWidth + 1):  # row
            patch0[j + halfWidth, i + halfWidth] = abs(img[centralCPy + j, centralCPx + i])
            patch1[j + halfWidth, i + halfWidth] = abs(img[centralCPy + j, (centralCPx - cellWidth) + i])
            patch2[j + halfWidth, i + halfWidth] = abs(img[(centralCPy - cellWidth) + j, (centralCPx - cellWidth) + i])
            patch3[j + halfWidth, i + halfWidth] = abs(img[(centralCPy - cellWidth) + j, centralCPx + i])
            patch4[j + halfWidth, i + halfWidth] = abs(img[(centralCPy - cellWidth) + j, (centralCPx + cellWidth) + i])
            patch5[j + halfWidth, i + halfWidth] = abs(img[centralCPy + j, (centralCPx + cellWidth) + i])
            patch6[j + halfWidth, i + halfWidth] = abs(img[(centralCPy + cellWidth) + j, (centralCPx + cellWidth) + i])
            patch7[j + halfWidth, i + halfWidth] = abs(img[(centralCPy + cellWidth) + j, centralCPx + i])
            patch8[j + halfWidth, i + halfWidth] = abs(img[(centralCPy + cellWidth) + j, (centralCPx - cellWidth) + i])

    meanPatch0 = np.mean(patch0)
    meanPatch1 = np.mean(patch1)
    meanPatch2 = np.mean(patch2)
    meanPatch3 = np.mean(patch3)
    meanPatch4 = np.mean(patch4)
    meanPatch5 = np.mean(patch5)
    meanPatch6 = np.mean(patch6)
    meanPatch7 = np.mean(patch7)
    meanPatch8 = np.mean(patch8)

    return meanPatch0, meanPatch1, meanPatch2, meanPatch3, meanPatch4, meanPatch5, meanPatch6, meanPatch7, meanPatch8

def candidate_Flag(candidatePosi, candidateLeni, cellWidthRefine, halfWidthRefine, imgWidth, SSLyi, imgdx, imgdy):
    targetFlag = False
    if candidatePosi < cellWidthRefine + halfWidthRefine or candidatePosi > imgWidth - cellWidthRefine - halfWidthRefine - 1:
        targetFlag = False
    else:
        if candidatePosi + (candidateLeni - 1) >= cellWidthRefine + halfWidthRefine and candidatePosi + (candidateLeni - 1) <= imgWidth - cellWidthRefine - halfWidthRefine - 1:
            mx0, mx1, mx2, mx3, mx4, mx5, mx6, mx7, mx8 = LCV(imgdx, cellWidthRefine, candidatePosi + (candidateLeni - 1), SSLyi - (candidateLeni - 1))
            my0, my1, my2, my3, my4, my5, my6, my7, my8 = LCV(imgdy, cellWidthRefine, candidatePosi + (candidateLeni - 1), SSLyi - (candidateLeni - 1))
        else:
            mx0, mx1, mx2, mx3, mx4, mx5, mx6, mx7, mx8 = LCV(imgdx, cellWidthRefine, candidatePosi, SSLyi - (candidateLeni - 1))
            my0, my1, my2, my3, my4, my5, my6, my7, my8 = LCV(imgdy, cellWidthRefine, candidatePosi, SSLyi - (candidateLeni - 1))
        max28dx = np.max([mx2, mx8])
        max24dy = np.max([my2, my4])
        if mx0 > max28dx and my0 > max24dy:
            targetFlag = True

    return targetFlag

def Gradient_refine(originalImg, candidatePosi, candidateLeni, SSLyi):
    _, imgWidth = originalImg.shape
    imgdx = cv2.Sobel(originalImg, cv2.CV_64F, 1, 0, ksize=3)
    imgdy = cv2.Sobel(originalImg, cv2.CV_64F, 0, 1, ksize=3)
    cellWidthRefine = 3
    halfWidthRefine = int(np.floor(cellWidthRefine / 2))
    if candidateLeni <= 2:
        cellWidthRefine = 3
        halfWidthRefine = int(np.floor(cellWidthRefine / 2))
    elif candidateLeni <= 4:
        cellWidthRefine = 5
        halfWidthRefine = int(np.floor(cellWidthRefine / 2))
    elif candidateLeni <= 6:
        cellWidthRefine = 7
        halfWidthRefine = int(np.floor(cellWidthRefine / 2))
    elif candidateLeni <= 8:
        cellWidthRefine = 9
        halfWidthRefine = int(np.floor(cellWidthRefine / 2))

    targetFlag = candidate_Flag(candidatePosi, candidateLeni, cellWidthRefine, halfWidthRefine, imgWidth, SSLyi, imgdx, imgdy)

    return targetFlag

def candidate_extract(originalImg, SSLy_list, threshold, cellWidth):
    _, imgWidth = originalImg.shape
    candidate = np.zeros((1, imgWidth))
    ratioSum = 0.0
    ratioNum = 0
    halfCellWidth = int(cellWidth/2)

    for x in range((cellWidth+halfCellWidth), imgWidth-(cellWidth+halfCellWidth)):
        y = SSLy_list[x] - 2
        m0, m1, m2, m3, m4, m5, m6, m7, m8 = LCV(originalImg, cellWidth, x, y)
        min12345 = np.min([m1, m2, m3, m4, m5])
        min678 = np.min([m6, m7, m8])
        max678 = np.max([m6, m7, m8])
        min678 = 0.1 * min678
        max678 = 0.1 * max678
        if min678 == 0:
            min678 = 1
        if m0 < min12345 and m0 > max678:
            ratio = abs(m0 - min12345) / abs(m0 - min678)
            ratioSum = ratioSum + ratio
            ratioNum = ratioNum + 1
            candidate[0, x] = ratio
    if ratioNum == 0:
        ratioNum = ratioNum + 1
    meanRatio = ratioSum / ratioNum

    for col in range(imgWidth):
        if candidate[0, col] >= threshold * meanRatio:
            candidate[0, col] = 1
        else:
            candidate[0, col] = 0
    _, candidateListLength = candidate.shape

    candidatePos = []
    candidateLen = []

    for i in range(candidateListLength):
        if i == 0 and candidate[0, i] == 1:
            candidatePos.append(i)
        elif i <= 382 and candidate[0, i] == 0 and candidate[0, i+1] == 1:
            candidatePos.append(i+1)

    for pos in candidatePos:
        if pos != 383:
            lengthTemp = 0
            while pos != 384 and candidate[0, pos] == 1:
                lengthTemp = lengthTemp + 1
                pos = pos + 1
            candidateLen.append(lengthTemp)
        elif pos == 383:
            candidateLen.append(1)

    return candidatePos, candidateLen

def target_refine(originalImg, candidatePos, candidateLen, SSLy_list):
    candidateNum = len(candidatePos)
    targetPos = []
    for i in range(candidateNum):
        candidatePosi = candidatePos[i]
        candidateLeni = candidateLen[i]
        SSLyi = SSLy_list[candidatePosi]
        targetFlag = Gradient_refine(originalImg, candidatePosi, candidateLeni, SSLyi)
        if targetFlag == True:
            if candidateLeni <= 2:
                candidatePosi = candidatePosi + 1
            elif candidateLeni <= 4:
                candidatePosi = candidatePosi + 2
            else:
                candidatePosi = candidatePosi + int(np.ceil(candidateLeni/2))

            targetPos.append([candidatePosi, SSLy_list[candidatePosi]])

    return targetPos

def target_detection(t, outputs, SSL_target_loader, idx, threshold, cellWidth):
    img, path = SSL_target_loader.dataset.getImgPath(idx)
    img = (img[0].cpu().numpy()) * 255
    height, width = img.shape
    TargetImgNameList, targets_anno = targets_anno_file('./jsonFile/S1.json')

    imgName = (path.split('/'))[-1]
    imgName_no_PNG = imgName[:-4]

    if imgName in TargetImgNameList:
        t0 = time()

        target_anno_idx = TargetImgNameList.index(imgName)
        target_anno = targets_anno[target_anno_idx]
        gt_targets = target_anno['point_targets']

        outputs, extra_outputs = outputs
        pred = outputs[0].cpu().numpy()
        pred_left = [0, 144]
        pred_right = [383, 144]

        for i, Horizon_coordinate in enumerate(pred):
            Horizon_coordinate = Horizon_coordinate[1:]  # remove conf
            lower, upper = Horizon_coordinate[0], Horizon_coordinate[1]  # left_endpoint, right_endpoint
            pred_left = (0, round(lower * height))
            pred_right = (383, round(upper * height))

        SSLy_list = SSLy(pred_left, pred_right)

        candidateStartPos, candidateLen = candidate_extract(img, SSLy_list, threshold, cellWidth)
        targetPos = target_refine(img, candidateStartPos, candidateLen, SSLy_list)

        # draw_detected_image(img, imgName_no_PNG + '_Detected', targetPos, 'lime')
        time_consume_one_frame = time() - t0
        time_consume_one_frame = time_consume_one_frame + t

        TD, FD, TU, td_diff = eval(targetPos, gt_targets)

        return TD, FD, TU, td_diff, len(targetPos), len(gt_targets), time_consume_one_frame
    else:
        return 0, 0, 0, 0, 0, 0, 0