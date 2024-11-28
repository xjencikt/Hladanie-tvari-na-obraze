import csv
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area

    return iou

def load_boxes(file_path):
    boxes = defaultdict(list)
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            name = row[0]
            boxes[name].append(list(map(int, row[2:])))
    return boxes

ground_truth_boxes = load_boxes('txt_files/true_original_rectangles.csv')
predicted_boxes = load_boxes('txt_files/mtcnn_sharpening.csv')

total_tp = total_fp = total_fn = 0

precisions = []
recalls = []
f1_scores = []

unique_names = set(ground_truth_boxes.keys()) | set(predicted_boxes.keys())
for name in unique_names:
    gt_coords = [box for box in ground_truth_boxes.get(name, [])]
    pred_coords = [box for box in predicted_boxes.get(name, [])]

    tp = fp = fn = 0
    matched_indices = set()

    for pred_index, pred_box in enumerate(pred_coords):
        if pred_box == [0, 0, 0, 0]:
            fn += 1
            continue

        matched = False
        for gt_index, gt_box in enumerate(gt_coords):
            iou = calculate_iou(pred_box, gt_box)
            if iou > 0.5 and gt_index not in matched_indices:
                tp += 1
                matched_indices.add(gt_index)
                matched = True
                break
        if not matched and pred_box != [0,0,0,0]:
            fp += 1



    total_tp += tp
    total_fp += fp
    total_fn += fn

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print(f"Metrics for {name}:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

total_precision = sum(precisions) / len(precisions) if len(precisions) > 0 else 0
total_recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 0
total_f1_score = sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else 0

print("\nTotal Metrics:")
print(f"Total True Positives (TP): {total_tp}")
print(f"Total False Positives (FP): {total_fp}")
print(f"Total False Negatives (FN): {total_fn}")
print(f"Total Precision: {total_precision}")
print(f"Total Recall: {total_recall}")
print(f"Total F1 Score: {total_f1_score}")
