import numpy as np


class Analyzer:
    """ Класс Analyzer сожержит все доступные методы анализа нарушений системы.
    """
    def __init__(self, config):
        """ Инициализация объекта класса

        Args:
            config (dict): чать конфигурационного файла системы, содержащая информацию
            об испольщуемых методах анализа нарушений
        """

        self.config = config
        self.violations = []  # список зафиксированных нарушений

    @staticmethod
    def _get_center(bbox):
        """вспомогательный метод для получения координат центра найденной области.

        Args:
            bbox (np.array): область, полученная моделью детекции
        Returns:
            np.array: центр области 
        """
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    
    @staticmethod
    def convert_time(millis):
        """метод для перевода временных меток к формату минуты:секунды

        Args:
            millis (float): временная метка

        Returns:
            str: временная метка в новом формате
        """
        millis = int(millis)
        seconds=(millis/1000)%60
        seconds = int(seconds)
        minutes=(millis/(1000*60))%60
        minutes = int(minutes)
        return f'{minutes}:{seconds:02}'

    @staticmethod
    def _split_violations(phone_usage_frames):
        """создает словарь, содержащий информацию о нарушениях 
           по каждому человеку в кадре

        Args:
            phone_usage_frames (list): список кадров на которых были зафиксированы нарушения

        Returns:
            dict: словарь с нарушениями 
        """
        person_violations = {}
        for frame_id, timestamp, person in phone_usage_frames:
            if person not in person_violations.keys():
                person_violations[person] = []
            person_violations[person].append((frame_id, timestamp))
        return person_violations
    
    def _get_phone_usage_frames(self, det, pos):
        """ метод для нахождения всех кадров где был использован телефон.

        Args:
            det (list): данные, полученные с помощью моделей детекции
            pos (list): данные, полученные с помощью моделей определения позы

        Returns:
            list: список кадров на которых были зафиксированы нарушения
        """
        phone_usage_frames = []
        max_wrist_dist = self.config['methods']['wrist_usage']['max_wrist_dist']

        for frame_id, timestamp ,detections in det:
            for obj_name, bbox in detections:
                if obj_name in self.config['methods']['wrist_usage']['detected_classes']:
                    for person, keys, _ in pos[frame_id-1][2]:
                        for wrist in keys[9:11]:
                            if np.linalg.norm(self._get_center(bbox) - wrist) < max_wrist_dist:
                                phone_usage_frames.append((frame_id, timestamp, person)) # + person
                                break
        return phone_usage_frames

    def wrist_usage(self, unprocessed_data):
        """С помощью данных, полученных из испольщуемых моделей обработки
           фиксирует временные промежутки, где использовался телефон. 

        Args:
            unprocessed_data (list): данные полученные из обработчика

        Returns:
            list: список нарушений
        """
        detection_data, pos_est_data = unprocessed_data
        method_config = self.config['methods']['wrist_usage']
        min_duratuin = method_config['min_duration']
        max_short_diff = method_config['max_short_diff']
        max_long_diff = method_config['max_long_diff']
        violations = []

        phone_usage_frames = self._get_phone_usage_frames(detection_data, pos_est_data)
        persons_violations = self._split_violations(phone_usage_frames)

        for person in persons_violations.keys():
            passed_time = 0
            last_stamp = 0
            start = 0

            for index, data in enumerate(persons_violations[person]):
                timestamp = data[1]
                diff = timestamp - last_stamp
                if start == 0:
                    start = timestamp
                elif (passed_time <= min_duratuin and diff <= max_short_diff) or (passed_time >= min_duratuin and diff <= max_long_diff):
                    passed_time = timestamp - start
                    last_stamp = timestamp
                    if index == len(persons_violations[person]) - 1:
                        violations.append([self.convert_time(start), self.convert_time(timestamp), person, 'использование телефона'])
                else:
                    if (passed_time > min_duratuin):
                        violations.append([self.convert_time(start), self.convert_time(timestamp), person, 'использование телефона'])
                    passed_time = 0
                    last_stamp = 0
                    start = 0
                    
        return violations
    
    def violation_analysis(self, data, methods):
        """метод для запуска анализаторов

        Args:
            data (dict): данные полученные из обработчика
            methods (list): список выбранных методов анализа

        Returns:
            list: список нарушений
        """
        for method in methods:
            func = getattr(self, method)
            unprocessed_data = []
            for dtype in self.config['methods'][method]['required_data']:
                unprocessed_data.append(data[dtype])
            self.violations.extend(func(unprocessed_data))
        return self.violations

    def clear_data(self):
        """удаление информации о найденных нарушениях"""
        self.violations = []
