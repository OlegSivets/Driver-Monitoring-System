import argparse
import pandas as pd

from dms import Engine
from dms.utils import Analyzer


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--detection-model', help="Путь к весам модели детекции телефонов",
                    type=str, default='trained_models/yolov8m.pt')
parser.add_argument('-p', '--pose-model', help="Путь к весам модели детекции поз",
                    type=str, default='trained_models/yolov8m-pose.pt')
parser.add_argument('-v', '--video-path', help="Путь к исходному видео",
                    type=str, default='data/test_video_cut.mp4')
parser.add_argument('-o', '--out-path', help="Путь для сохранения обработанного видео",
                    type=str, default='results/processed_videos/test_result.avi')
parser.add_argument('-c', '--csv-save-path', help="Путь для сохранения результата работы модели", 
                    type=str, default='results/res_data.csv')

args = parser.parse_args()

model = Engine()
model.load_models(args.detection_model, args.pose_model)
model.process_video(args.video_path, args.out_path)

analyzer = Analyzer()
violations = analyzer.process_model_data(model.pos_est_data, model.detected_data)
ans = analyzer.process_violations(violations)


ans_df = pd.DataFrame(ans, columns=['Start', 'End'])
ans_df.to_csv(args.csv_save_path, index=False)

print(ans_df)
