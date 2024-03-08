import time
import torch
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from ultralytics import YOLO
from dms.utils import VideoRenderer


class Engine:
    """_summary_
    """
    def __init__(self, batch_size=8, grayscale_adapted=False):
        self.detected_history = []
        self.pos_est_values = []
        self.detected_values = []
        self.sec_per_frame = None
        self.batch_size = batch_size
        self.grayscale_adapted = grayscale_adapted

        # models
        self.detect_model = (None,)
        self.pos_est_model = None

        # video parameters
        self.video_type = "avi"  # saved video type

    def load_models(self, detection_modelname=None, pose_estimator_name=None):
        """_summary_

        Args:
            detection_modelname (_type_, optional): _description_. Defaults to None.
            pose_estimator_name (_type_, optional): _description_. Defaults to None.
        """

        if torch.cuda.is_available():
            if detection_modelname:
                self.detect_model = YOLO(detection_modelname)
                self.detect_model.to("cuda")
            if pose_estimator_name:
                self.pos_est_model = YOLO(pose_estimator_name)
                self.pos_est_model.to("cuda")

    def handle_detection(self, results, frames, timestamps, frame_ids, save=False):
        """_summary_

        Args:
            results (_type_): _description_
            frames (_type_): _description_
            timestamps (_type_): _description_
            frame_ids (_type_): _description_
            save (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        for i, res in enumerate(results):
            detected_classes = []
            detected_positions = []
            boxes = res.boxes.cpu().numpy()

            for box in boxes:
                class_name = list(self.detect_model.names)[int(box.cls)]
                detected_classes.append(str(class_name))
                xyxy = box.xyxy[0]

                if save:
                    confidence = str(round(box.conf[0].item(), 2))
                    label = f"{class_name}: {confidence}"
                    frames[i] = VideoRenderer.plot_boxes(frames[i], xyxy, label)
                detected_positions.append(xyxy.tolist())

            if detected_classes:
                self.detected_values.append(
                    [frame_ids[i], timestamps[i], detected_positions]
                )
        return frames

    def handle_pos_est(self, results, frames, timestamps, frame_ids, save=False):
        """_summary_

        Args:
            results (_type_): _description_
            frames (_type_): _description_
            timestamps (_type_): _description_
            frame_ids (_type_): _description_
            save (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        for i, result in enumerate(results):
            keypoints = result.keypoints.xy.cpu().numpy().tolist()
            for pose in keypoints:
                if save:
                    for point in pose:
                        frames[i] = cv2.circle(
                            frames[i],
                            (int(point[0]), int(point[1])),
                            radius=0,
                            color=(0, 255, 0),
                            thickness=10,
                        )
            self.pos_est_values.append(
                [frame_ids[i], timestamps[i], len(results[i]), keypoints]
            )
        return frames

    def process_batch(self, frames, timestamps, frame_ids, save):
        """_summary_

        Args:
            frames (_type_): _description_
            timestamps (_type_): _description_
            frame_ids (_type_): _description_
            save (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.pos_est_model is not None:
            pos_est_results = self.pos_est_model(frames, verbose=False)
            frames = self.handle_pos_est(
                pos_est_results, frames, timestamps, frame_ids, save
            )
        if self.detect_model is not None:
            conf = 0.8
            if self.grayscale_adapted and np.array(frames).std(axis=-1).mean() < 2:
                conf = 0.4
            detection_results = self.detect_model(frames, verbose=False, conf=conf)
            frames = self.handle_detection(
                detection_results, frames, timestamps, frame_ids, save
            )
        return frames

    def process_video(
        self, data_path, out_path, detection=True, pos_estimation=True, save=False
    ):
        
        """_summary_

        Args:
            data_path (_type_): _description_
            out_path (_type_): _description_
            detection (bool, optional): _description_. Defaults to True.
            pos_estimation (bool, optional): _description_. Defaults to True.
            save (bool, optional): _description_. Defaults to True.
        """

        cap = cv2.VideoCapture(data_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        self.sec_per_frame = 1 / fps
        self.detected_values = []
        self.pos_est_values = []
        self.detected_history = []
        self.use_detection = detection
        self.use_pose_estimation = pos_estimation

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_width = int(cap.get(3))

        frame_height = int(cap.get(4))

        if save:
            codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")  # disable=no-member
            out = cv2.VideoWriter(out_path, codec, fps, (frame_width, frame_height))

        start = time.time()
        with tqdm(total=frame_count) as pbar:
            while cap.isOpened():
                frames = []
                frame_ids = []
                timestamps = []
                for _ in range(self.batch_size):
                    success, frame = cap.read()
                    if not success:
                        break
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                    frame_ids.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    frames.append(frame)
                if len(frames) != 0:
                    frames = self.process_batch(frames, timestamps, frame_ids, save)
                    if save:
                        for frame in frames:
                            out.write(frame)
                if not success:
                    break
                pbar.update(len(frames))
        end = time.time() - start
        print(f"Time: {end}")
        self.pos_est_data = pd.DataFrame(
            self.pos_est_values,
            columns=["Frame_id", "Time", "People_count", "KeyPoints"],
        )
        self.detected_data = pd.DataFrame(
            self.detected_values, columns=["Frame_id", "Time", "Position"]
        )

    def get_detected_data(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.detected_data

    def clear_data(self):  # clear detected_data
        """_summary_"""
        self.detected_data = self.detected_data.iloc[0:0]
