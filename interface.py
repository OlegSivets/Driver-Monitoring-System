import gradio as gr
import pandas as pd
import cv2

from dms.engine import Engine


inputs = [
    gr.Video(label='Input Video'),
    gr.Textbox(lines=1, placeholder="0:00", label='Введите время нарушения')
]

outputs = [
    gr.Image(type='numpy'),
    gr.Dataframe(
        label="Результат обработки видео",
        row_count=3,
        col_count=4,
        headers=['Начало', 'Конец', 'Человек', 'Нарушение'],
    ),
]

last_video = None
engine = Engine()

def logic(video_path, image_time):
    global last_video
    global pd_ans
    image = None

    if video_path and last_video != video_path:
        violations = engine.violations_search(video_path)
        pd_ans = pd.DataFrame(violations, columns=['Начало', 'Конец', 'Человек', 'Нарушение'])
        last_video = video_path

    if image_time and video_path:
        time_ms = sum([int(el) * 60 ** i for i, el in enumerate(image_time.split(':')[::-1])]) * 1000
        image = engine.show_violations(time_ms, video_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image, pd_ans


demo = gr.Interface(
    logic,
    inputs=inputs,
    outputs=outputs,
)

demo.launch()
