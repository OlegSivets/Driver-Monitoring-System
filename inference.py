import argparse
import pandas as pd

from dms.engine import Engine

parser = argparse.ArgumentParser()

parser.add_argument('-v', '--video-path', help="Путь к исходному видео",
                    type=str, default='data/test_short.mp4')
parser.add_argument('-s', '--save-path', help="Место для хранения csv",
                    type=str, default='results/results.csv')
args = parser.parse_args()

dms = Engine()
violations = dms.violations_search(args.video_path)

ans_df = pd.DataFrame(violations, columns=['Начало', 'Конец', 'Человек', 'Нарушение'])
ans_df.to_csv(args.save_path, index=False)
