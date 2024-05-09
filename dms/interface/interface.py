import gradio as gr
import pandas as pd
import cv2

from dms.engine import Engine


class Interface():
    """_summary_
    """
    def __init__(self, ) -> None:
        """_summary_
        """
        self.engine = Engine()

        self.inputs  = [
            gr.Video(label='Input Video'),
            gr.Textbox(lines=1, placeholder="0:00", label='Введите время нарушения')
        ]

        self.outputs = [
            gr.Image(type='numpy'),
            gr.Dataframe(
                label="Результат обработки видео",
                row_count=3,
                col_count=4,
                headers=['Начало', 'Конец', 'Человек', 'Нарушение'],
            ),
        ]

        self.demo = gr.Interface(
            self.logic,
            inputs=self.inputs,
            outputs=self.outputs,
        )

        self.last_video = None
        self.pd_data = None

    def logic(self, video_path, image_time):
        """_summary_

        Args:
            video_path (_type_): _description_
            image_time (_type_): _description_

        Returns:
            _type_: _description_
        """
        image = None

        if video_path and self.last_video != video_path:
            violations = self.engine.violations_search(video_path)
            self.pd_ans = pd.DataFrame(violations, columns=['Начало', 'Конец', 'Человек', 'Нарушение'])
            self.last_video = video_path

        if image_time and video_path:
            time_ms = sum([int(el) * 60 ** i for i, el in enumerate(image_time.split(':')[::-1])]) * 1000
            image = self.engine.show_violations(time_ms, video_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image, self.pd_ans


    def launch(self):
        """_summary_
        """
        self.demo.launch()

if __name__ == '__main__':
    interface = Interface()
    interface.launch()
