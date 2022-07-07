import os
import sys
import torch
import numpy as np
import os
import cv2
import numpy as np

# ----------------------------------------------
base_dir = os.getcwd()
tools_path = os.path.join(base_dir, 'tools')
yolo_module_path = os.path.join(base_dir, 'tools/yolov5')
sys.path.append(tools_path)
sys.path.append(yolo_module_path)
# ==============================================

from yolov5.models.experimental import attempt_load
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

class Detector():

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.half = False

    def load_model(self, weights: str, device: str = 'cuda') -> None:
        device = select_device(device)
        model = attempt_load(weights, map_location=device)
        half = device.type != 'cpu' # в зависимости от устройства преобразуем к соответствующему типу
        if half:
            model.half()

        self.model = model
        self.device = device
        self.half = half

    def load(self, path_to_model: str, device: str = 'cpu') -> None:
        if device == "gpu":
            device = "mps" # device = "cuda"
        else:
            device = "cpu"
        self.load_model(path_to_model, device)

    def draw_rectangle(self, output, image):
        
        thickness = 1 #толщина рамки
        font_scale = 1
        for box in output:
        # извлекаем координаты ограничивающего прямоугольника
            x0, y0 = int(box[0]), int(box[1])
            x1, y1 = int(box[2]), int(box[3])
            confidences = float(box[4])
            # рисуем прямоугольник ограничивающей рамки и подписываем на изображении
            color = [255,255,123] #поменять цвет
            cv2.rectangle(image, (x0, y0), (x1, y1), color=color, thickness=thickness)
            text = f"{'head'}: {confidences:.2f}"
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
    
    def detect_bbox(self,
                    img: np.ndarray,
                    img_size: int = 640,
                    stride: int = 32,
                    min_accuracy: float = 0.1):
        image = img
        img_shape = img.shape
        #resize и pad изображения, соблюдая ограничения stride-multiple
        img = letterbox(img, img_size, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        #возвращает Contiguous Array
        #это просто массив, хранящийся в непрерывном блоке памяти:
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        #to torch.float16 or float
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) # -> [.,.,.,...]

        pred = self.model(img)[0]
        #Non maximum suppression, возвращает лист размером (n,6) tensor per image [xyxy, conf, cls]
        pred = non_max_suppression(pred)
        res = []
        for i, det in enumerate(pred):
            if len(det):
                #Масштабирование координат (xyxy) с img.shape[2:] на img_shape
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_shape).round()
                res.append(det.cpu().detach().numpy())
        output = []
        if len(res):
            output = [[x1, y1, x2, y2, acc, b] for x1, y1, x2, y2, acc, b in res[0] if acc > min_accuracy]
        image = self.draw_rectangle(output, image)
        return image