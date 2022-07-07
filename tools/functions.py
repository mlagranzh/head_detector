import os
import time
import datetime
import cv2
import numpy as np

def start_processing_time():
    start_time_hms = datetime.datetime.now()
    print('Начало работы в {}:{:02d}:{:02d}'.format(start_time_hms.hour, start_time_hms.minute, start_time_hms.second))
    return time.time()

# ----------------------------------------------
def seconds_to_m_s(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return m, s
# ==============================================

def detect_heads(model_yolo, image, threshold=0.6):
    boxes = model_yolo(image).pandas().xyxy[0]
    thickness = 1 #что это ?? ?
    font_scale = 1
    labels_count = 100
    colors = [(9*i%255, 3*i%255, 6*i%255) for i in range(labels_count)]
    for i in range(len(boxes)):
        # извлекаем координаты ограничивающего прямоугольника
        x0, y0 = int(boxes.iloc[i].xmin), int(boxes.iloc[i].ymin)
        x1, y1 = int(boxes.iloc[i].xmax), int(boxes.iloc[i].ymax)
        class_ids = boxes.iloc[i]['class']
        name = boxes.iloc[i]['name']
        confidences = float(boxes.iloc[i]['confidence'])
        if (confidences < threshold):
            continue
        # рисуем прямоугольник ограничивающей рамки и подписываем на изображении
        color = [int(c) for c in colors[class_ids]]
        cv2.rectangle(image, (x0, y0), (x1, y1), color=color, thickness=thickness)
        text = f"{name}: {confidences:.2f}"
        # вычисляем ширину и высоту текста, чтобы рисовать прозрачные поля в качестве фона текста
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = x0
        text_offset_y = y0 - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        # добавить непрозрачность (прозрачность поля)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        # теперь поместите текст (метка: доверие%)
        cv2.putText(image, text, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
    
    return image
# ==============================================

def get_mode_torch() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "gpu"
    return "cpu"

# ----------------------------------------------
def select_device(device, cpu_restrict=False, cpu_cores=1):
    if device == 'auto':
        device = get_mode_torch()

    if device == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    else:
        device = 'cpu'

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        if cpu_restrict:
            import tensorflow as tf
            tf.config.threading.set_inter_op_parallelism_threads(cpu_cores)
            tf.config.threading.set_intra_op_parallelism_threads(cpu_cores)
            tf.config.set_soft_device_placement(True)

    return device
# ==============================================