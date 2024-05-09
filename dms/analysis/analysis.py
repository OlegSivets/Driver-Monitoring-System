import numpy as np


class Analyzer:
    """_summary_
    """
    def __init__(self, config):
        """_summary_

        Args:
            config (_type_): _description_
        """
        self.config = config
        self.violations = []

    @staticmethod
    def _get_center(bbox):
        """_summary_

        Args:
            bbox (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    
    @staticmethod
    def convert_time(millis):
        """_summary_

        Args:
            millis (_type_): _description_

        Returns:
            _type_: _description_
        """
        millis = int(millis)
        seconds=(millis/1000)%60
        seconds = int(seconds)
        minutes=(millis/(1000*60))%60
        minutes = int(minutes)
        return f'{minutes}:{seconds:02}'

    @staticmethod
    def _split_violations(phone_usage_frames):
        """_summary_

        Args:
            phone_usage_frames (_type_): _description_

        Returns:
            _type_: _description_
        """
        person_violations = {}
        for frame_id, timestamp, person in phone_usage_frames:
            if person not in person_violations.keys():
                person_violations[person] = []
            person_violations[person].append((frame_id, timestamp))
        return person_violations
    
    def _get_phone_usage_frames(self, det, pos):
        """_summary_

        Args:
            det (_type_): _description_
            pos (_type_): _description_

        Returns:
            _type_: _description_
        """
        phone_usage_frames = []
        max_wrist_dist = self.config['wrist_phone_usage']['max_wrist_dist']

        for frame_id, timestamp ,detections in det:
            for obj_name, bbox in detections:
                if obj_name == 'cell phones':
                    for person, keys, _ in pos[frame_id-1][2]:
                        for wrist in keys[9:11]:
                            if np.linalg.norm(self._get_center(bbox) - wrist) < max_wrist_dist:
                                phone_usage_frames.append((frame_id, timestamp, person)) # + person
                                break
        return phone_usage_frames

    def wrist_phone_usage(self, unprocessed_data):
        """_summary_

        Args:
            unprocessed_data (_type_): _description_

        Returns:
            _type_: _description_
        """
        detection_data, pos_est_data = unprocessed_data
        min_duratuin = self.config['wrist_phone_usage']['min_duration']
        max_short_diff = self.config['wrist_phone_usage']['max_short_diff']
        max_long_diff = self.config['wrist_phone_usage']['max_long_diff']
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
        """_summary_

        Args:
            data (_type_): _description_
            methods (_type_): _description_

        Returns:
            _type_: _description_
        """
        for method in methods:
            func = getattr(self, method)
            unprocessed_data = []
            for dtype in self.config[method]['required_data']:
                unprocessed_data.append(data[dtype])
            self.violations.extend(func(unprocessed_data))
        return self.violations

    def clear_data(self):
        """_summary_
        """
        self.violations = []
