import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask
import matplotlib.pyplot as plt
import cv2 as cv

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (list): [x1, y1, x2, y2] coordinates of the first box.
        box2 (list): [x1, y1, x2, y2] coordinates of the second box.

    Returns:
        float: IoU value between 0 and 1.
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def calculate_map(ground_truths, predictions, iou_threshold=0.5):
    """Calculates mean Average Precision (mAP) for object detection.

    Args:
        ground_truths (list): List of ground truth bounding boxes.
        predictions (list): List of predicted bounding boxes with confidence scores.
        iou_threshold (float): IoU threshold for considering a prediction as a true positive.

    Returns:
        float: mAP value.
    """

    # average_precisions = []

    # for class_label in set(gt['class'] for gt in ground_truths):
    # class_ground_truths = [gt for gt in ground_truths if gt['class'] == class_label]
    # class_predictions = [pred for pred in predictions if pred['class'] == class_label]

    # class_predictions.sort(key=lambda x: x['confidence'], reverse=True)

    # predictions.sort(key=lambda x: x['confidence'], reverse=True)

    true_positives = 0
    false_positives = 0
    precisions = []
    recalls = []

    for pred in predictions:
        iou_max = 0
        for gt in ground_truths:
            iou = calculate_iou(pred, gt)
            if iou > iou_max:
                iou_max = iou
        print(f'iou_max = {iou_max}')

        if iou_max >= iou_threshold:
            true_positives += 1
        else:
            false_positives += 1

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / len(ground_truths)

        precisions.append(precision)
        recalls.append(recall)

        print(f'precisions = {precisions}, recalls = {recalls}')

    if len(precisions) > 2:
        average_precision = 0
        for i in range(1, len(precisions)):
            average_precision += (recalls[i] - recalls[i - 1]) * precisions[i]
        return average_precision
    else:
        return precisions[0]

    # average_precisions.append(average_precision)

def get_ground_truth_ann(image_name=None, show=False):
    """
    Loads ground truth bounding boxes from a COCO-format JSON file and optionally displays them.

    Parameters
    ----------
    image_name : str, optional
        The name of the image (default is 'orange'). The function expects an annotation file 
        in './img/' named '<image_name>_annotation.json'.
    show : bool, optional
        If True, displays the image with annotated bounding boxes (default is False).

    Returns
    -------
    bboxes : list of lists
        Bounding boxes in [x_min, y_min, x_max, y_max] format.

    Example
    -------
    bboxes = get_ground_truth_ann('apple', show=True)
    """

    if image_name == None:
        image_name = 'orange'

    # Load the COCO dataset
    annotation_file = './coco-annotations/' + image_name + '_annotation.json'
    coco = COCO(annotation_file)

    # Get all image IDs and choose one image
    image_ids = coco.getImgIds()
    image_id = image_ids[0]

    # Load the image metadata
    image_info = coco.loadImgs(image_id)[0]
    h, w = image_info['height'], image_info['width']

    # Get annotations for the image
    ann_ids = coco.getAnnIds(imgIds=image_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    # print(anns[0]['bbox'])
    bboxes = []
    for ann in anns:
        bb = ann['bbox']
        bboxes.append([bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]])

    if show:
        img = cv.imread('./img/' + image_name + '.jpg')
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(img)
        coco.showAnns(anns, draw_bbox=True)

    return bboxes
