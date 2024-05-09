import time
import torch
import cv2
from tqdm import tqdm
from ultralytics import YOLO


class VideoHandler():
    """ Класс для обработки видео. 
        Применяет ряд моделей к видео и записывает результат обработки
    """

    def __init__(self, config) -> None:
        # конфигурация обработчика
        self.config = config

        # инициализированные модели 
        self.models = {
            'detection': [],
            'pos_est': []
        }

        # Информация полученная после обработки
        self.data = {
            'detection': [],
            'pos_est': []
        }

        self.cuda_status = torch.cuda.is_available()

        # Инициализация моделей обработки
        self.load_models()

    def load_models(self):
        # Инициализация моделей обработки
        init_models = []
        for model_name, model_conf in self.config['models'].items():

            if model_conf['format'] == 'YOLO':
                init_models.append(YOLO(model_conf['path']))
            else:
                # обработка прочих форматов моделей пока не поддерживается
                pass
            for model in init_models:
                if self.cuda_status:
                    model.to('cuda')
                self.models[model_conf['task']].append((model_name, model))

    # Метод для обработки результата модели детекции формата YOLO
    def handle_yolo_detection(self, model, frames, conf): 
        results = model(frames, verbose=False, conf=conf)  # TODO добавить conf в конфиг
        names = model.names
        detection_data = []

        for _, res in enumerate(results):
            detected_objects = []
            boxes = res.boxes.cpu().numpy()
            for box in boxes:
                class_name = names[int(box.cls)]
                xyxy = box.xyxy[0]
                detected_objects.append((class_name, xyxy.tolist()))
            
            if detected_objects:
                detection_data.append(detected_objects)
            else:
                detection_data.append([])
        return detection_data

    # Метод для обработки результата модели определения позы формата YOLO
    def handle_yolo_pos_est(self, model, frames, conf):
        results = model(frames, verbose=False, conf=conf)
        pos_est_data = []

        for _, res in enumerate(results):
            objects_pos = []
            if res.keypoints:
                boxes = res.boxes.cpu().numpy()
                keypoints = res.keypoints.xy.cpu().numpy().tolist()
                for i, box in enumerate(boxes):
                    xyxy = box.xyxy[0]
                    keys = keypoints[i]
                    objects_pos.append([i, keys, xyxy.tolist()])

            if objects_pos:
                pos_est_data.append(objects_pos)
            else:
                pos_est_data.append([])
        return pos_est_data

    # Метод для обработки набора кадров выбранными моделями и записи результата
    def process_batch(self, frames, timestamps, frame_ids): # TODO привести все к одному формату, чтобы код не дублировался

        detection_models = self.models['detection']
        pos_est_models = self.models['pos_est']
        batch_det_data = []
        batch_pos_data = []

        # Обработка кадров моделями детекции
        if detection_models:
            batch_det_data = [[frame_ids[i], timestamps[i], []] for i, _ in enumerate(frames)]
            for model_name, model in detection_models:
                det_data = None
                # Обработка моделей формата YOLO
                if self.config['models'][model_name]['format'] == 'YOLO':
                    conf = self.config['models'][model_name]['specific_params']['conf']
                    det_data = self.handle_yolo_detection(model, frames, conf)

                if det_data:
                    for i, _ in enumerate(frames):
                        batch_det_data[i][2].extend(det_data[i])
        
        # Обработка кадров моделями определения позы
        if pos_est_models:
            batch_pos_data = [[frame_ids[i], timestamps[i], []] for i, _ in enumerate(frames)]
            for model_name, model in pos_est_models:
                pos_est_data = None
                # Обработка моделей формата YOLO
                if self.config['models'][model_name]['format'] == 'YOLO':
                    conf = self.config['models'][model_name]['specific_params']['conf']
                    pos_est_data = self.handle_yolo_pos_est(model, frames, conf)
                if pos_est_data:
                    for i, _ in enumerate(frames):
                        batch_pos_data[i][2].extend(pos_est_data[i])

        # Запись результатов обработки
        self.data['detection'].extend(batch_det_data)
        self.data['pos_est'].extend(batch_pos_data)


    def process_video(self, video_path):
        batch_size = self.config['processing']['BATCH_SIZE']
        
        # Информация о видео
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Цикл обработки видео
        start = time.time()
        with tqdm(total=frame_count) as pbar:
            while cap.isOpened():
                frames = []
                frame_ids = []
                timestamps = []
                # Собираем кадры в батч для обработки
                for _ in range(batch_size):
                    success, frame = cap.read()
                    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                    if not success:
                        break
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                    frame_ids.append(frame_id)
                    frames.append(frame)

                if len(frames) != 0:
                    # Отправляем собранный батч на обработку
                    self.process_batch(frames, timestamps, frame_ids)

                if not success:
                    break
                pbar.update(len(frames))
        end = time.time() - start
        print(f"Time: {end}")

        return self.data

    def get_frame_data(self, timestamp, task):
        for _, time, obj_data in self.data[task]:
            if time >= timestamp:
                return obj_data

    def clear_data(self):
        for key in self.data.keys():
            self.data[key] = []
        