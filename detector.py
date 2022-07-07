import os
import cv2
import torch
import time
import pickle
import shutil
from tqdm import tqdm
import numpy as np
from tools.functions import (detect_heads, seconds_to_m_s, start_processing_time, select_device)
from tools.YoloV5Detector import Detector

#ограничить потребление ресурсов процесса:
# pid = os.getpgid(0)
# res = os.system(f"cpulimit -l 2 -p {pid} &")

start_work = time.time()

# ----------------------------------------------
#            ***  DIRECTORY  ***
# Путь к папке с входными видео.
videos_dir = "video"
# Название текущего видео
video_name = '1.mp4'
# Путь к папке с выходными видео
# (создастся автоматически).
result_dir = 'result'

use_device = ('cpu', 'gpu')[0]

# Порого вероятности, ниже которой
# предсказание отфильтровывается.
SCORE_THRESH = 0.1

# Отображать видео обработки в реальном времени
real_time_video = True

# Сколько кадров должно пройти перед
# сохранением контрольной точки
# с учётом пропуска кадров.
checkpoint_size = 100

# Начать с начала, даже если 
# есть контрольная точка.
RESET = False

# ----------------------------------------------
video_name_short = video_name.split('.')[0]

print('\nВидео:', video_name)
video_in_path = os.path.join(videos_dir, video_name)
cap = cv2.VideoCapture(video_in_path)

vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Ширина и высота кадра:', vid_width, vid_height)

vid_fps = int(cap.get(cv2.CAP_PROP_FPS))
print('Количество кадров в секунду:', vid_fps)

vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Всего кадров в видео:', vid_length)

length_sec = vid_length/vid_fps
print('Длительность видео в секундах: {:.2f}'.format(length_sec))
# ==============================================
total_save = int(vid_length)
print('Всего получится изображений из видео:', total_save)
# ==============================================

# detector = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True, force_reload=True)
# detector.to(torch.device(use_device))

detector = Detector()
detector.load("Model_data/yolov5.pt", use_device)

# ----------------------------------------------
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

data_out_dir = os.path.join(result_dir, video_name_short)
if not os.path.exists(data_out_dir):
    os.mkdir(data_out_dir)
else:
    if RESET:
        shutil.rmtree(data_out_dir)
        os.mkdir(data_out_dir)

# ----------------------------------------------    
info_pickles_path = os.path.join(data_out_dir, '{}_info.p'.format(video_name_short))

if os.path.exists(info_pickles_path):
    with open(info_pickles_path, 'rb') as file:
        info_pickles = pickle.load(file)

    video_name_pickle = info_pickles[0]
    frame_ind = info_pickles[1]
    frame_ind_load = info_pickles[1]

    assert video_name_pickle == video_name, 'Wrong video name'

    assert frame_ind != 'done', 'Video already processed'

    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)

    print('\nЗагрузка с контрольной точки!')
else:
    frame_ind = 0
    frame_ind_load = 0
    print('\nЗагрузка с нуля!')
# ==============================================

# ----------------------------------------------
processing_start = start_processing_time()
print('\nВыполняется обработка и сохранение кадров...')

output_video = cv2.VideoWriter(data_out_dir+"/file.avi", cv2.VideoWriter_fourcc('M','J','P','G'), vid_fps, (vid_width,vid_height))
fps_per_frame = []

for i in tqdm(range(frame_ind, vid_length)):
    start_time = time.time()
    ret, image = cap.read()

    if not ret:
        break
    bboxes = detector.detect_bbox(image, min_accuracy=SCORE_THRESH)
    # bboxes = detect_heads(detector, image, threshold=SCORE_THRESH)

    output_video.write(bboxes)

    frame_ind += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
    
    #считаем количество fps обработки
    fps_per_frame.append(1. / (time.time() - start_time))

    if frame_ind % (checkpoint_size) == 0:

        info_pickles = (video_name, frame_ind)       
        
        with open(info_pickles_path, 'wb') as file:
            pickle.dump(info_pickles, file)

        print("Текущее fps обработки: ", np.mean(fps_per_frame), '\n')


    if real_time_video:
        cv2.imshow('VIDEO', bboxes)
        cv2.waitKey(1)
            
cap.release()
cv2.destroyAllWindows()
print('Сохранение кадров завершено.')
# ==============================================

# ----------------------------------------------
info_pickles = (video_name, 'done')         

with open(info_pickles_path, 'wb') as file:
    pickle.dump(info_pickles, file)
    
duration = time.time()-start_work
processing_duration = time.time()-processing_start
print('\nВсего обработка видео заняла {} минут и {} секунд'.format(*seconds_to_m_s(duration)))
print('Точнее:', round(duration, 2))
print('Только обработка:', round(processing_duration, 2))
# ==============================================
