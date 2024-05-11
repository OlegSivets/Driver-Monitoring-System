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
        """Инициализация объекта класса

        Args:
            config (_type_): чать конфигурационного файла системы, содержащая информацию
            об используемых моделях и параметрах обработки видео
        """

        self.config = config
        self.models = {
            'detection': {},
            'pos_est': {}
        }
        self.data = {
            'detection': [],
            'pos_est': []
        }

        self.cuda_status = torch.cuda.is_available()
        self.active_models = []
        self.load_models()

    def load_models(self):
        """метод для инициализации моделей обработки
        """
        init_models = []
        for model_name, model_conf in self.config['models'].items():
            if model_conf['format'] == 'YOLO':
                init_models.append(YOLO(model_conf['path']))
            for model in init_models:
                if self.cuda_status:
                    model.to('cuda')
                self.models[model_conf['task']][model_name] = model

    def handle_yolo_detection(self, model, frames, conf):
        """метод для обработки результата модели детекции формата YOLO

        Args:
            model (_type_): модель детекции
            frames (np.array): кадры для обработки
            conf (float): параметр уверенности модели

        Returns:
            list: результат обработки
        """
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

    def handle_yolo_pos_est(self, model, frames, conf):
        """етод для обработки результата модели определения позы формата YOLO

        Args:
            model (_type_): модель определения позы
            frames (np.array): кадры для обработки 
            conf (float): параметр уверенности модели

        Returns:
            list: результат обработки
        """
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
    def process_batch(self, frames, timestamps, frame_ids):
        """метод для обработки набора кадров с помощью выбранных моделей и записи полученой информации 

        Args:
            frames (mp.array): кадры для обработки 
            timestamps (list): список с временными метками кадров 
            frame_ids (list): список с id кадров 
        """

        detection_models = []
        pos_est_models = []
        for model_name in self.active_models:
            model_config = self.config['models'][model_name]
            if model_config['task'] == 'detection':
                model = self.models['detection'][model_name]
                detection_models.append((model_name, model))
            elif model_config['task'] == 'pos_est':
                model = self.models['pos_est'][model_name]
                pos_est_models.append((model_name, model))

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


    def process_video(self, video_path, active_models):
        """метод обработки видео

        Args:
            video_path (str): путьк видео

        Returns:
            dict: результат обработки видео выбранными моделями
        """
        batch_size = self.config['processing']['BATCH_SIZE']
        self.active_models = active_models
        
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
        """метд для полученния информации по обработанному кадру

        Args:
            timestamp (float): временная метка кадра
            task (str): тип обработки

        Returns:
            _type_: _description_
        """
        for _, time, obj_data in self.data[task]:
            if time >= timestamp:
                return obj_data

    def clear_data(self):
        """удаление информации об обработанных кадрах"""
        for key in self.data.keys():
            self.data[key] = []
        