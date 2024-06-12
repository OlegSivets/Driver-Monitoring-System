import gradio as gr
import pandas as pd
import cv2

from dms.engine import Engine
from dms.settings import config


class Interface():
    """Класс демонстрационного интерфейса системы
    """
    def __init__(self) -> None:
        """Инициализация объекта класса"""
        self.engine = Engine()

        self.inputs  = [
            gr.Video(label='Input Video', height=400),
            gr.Textbox(lines=1, placeholder="0:00", label='Введите время нарушения'),
            gr.Dropdown(
                choices= list(config['handler']['models'].keys()), 
                value=config['handler']['default_models'],
                multiselect=True, label="Методы обработки видео", info="Доступные модели обработки видео"
            ),
            gr.Dropdown(
                choices=list(config['analyser']['methods'].keys()),  # ["Использование телефона"]
                value=config['analyser']['default_methods'],  # ["Использование телефона"]
                multiselect=True, label="Методы анализа", info="Доступные методы анализа нарушений"
            ),
        ]

        self.outputs = [
            gr.Image(type='numpy', height=400),
            gr.Dataframe(
                label="Результат обработки видео",
                row_count=7,
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

    def logic(self, video_path, image_time, models, methods):
        """метод, отвечающий за функционал всех компонентов интерфейса

        Args:
            video_path (str): путь к видео
            image_time (str): временная метка

        Returns:
            (nd.array, pd.Dataframe): обработанный кадр, информация о нарушениях
        """
        image = None
        methods = ['wrist_usage']
        if video_path and self.last_video != video_path:
            violations = self.engine.violations_search(video_path, models, methods)
            self.pd_ans = pd.DataFrame(violations, columns=['Начало', 'Конец', 'Человек', 'Нарушение'])
            self.last_video = video_path

        if image_time and video_path:
            time_ms = sum([int(el) * 60 ** i for i, el in enumerate(image_time.split(':')[::-1])]) * 1000
            image = self.engine.show_violations(time_ms, video_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image, self.pd_ans


    def launch(self):
        """запуск интерфейса"""
        self.demo.launch()

if __name__ == '__main__':
    interface = Interface()
    interface.launch()
