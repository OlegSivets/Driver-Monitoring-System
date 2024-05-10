from dms.analysis import Analyzer
from dms.handler import VideoHandler
from dms.utils import VideoRenderer
from dms.settings import config

class Engine:
    """
    Класс Engine является связующим звеном между обработчиком и анализаторм.

    """
    def __init__(self, config = config): 
        """""
        Инициализация объекта класса
        Args:
            config (dict): конфигурация запуска системы содержит информацию
            о моделях, методах поиска нарушений и параметрах обработки видео.
            Defaults to config.
        """""

        self.config = config
        self.handler = VideoHandler(self.config['handler'])
        self.analizer = Analyzer(self.config['analyser'])
        self.renderer = VideoRenderer()

    def violations_search(self, video_path, methods=None):
        """метод для поиска нарушений на видео. 

        Args:
            video_path (str): путь к видео
            methods (list, optional): выбраные методы анализа видео, при отсутствии
            параметра будут использованы все доступные методы

        Returns:
            list: список нарушений
        """
        self.handler.clear_data()
        self.analizer.clear_data()

        if not methods:
            methods = list(self.config['analyser'].keys())

        model_process_res = self.handler.process_video(video_path)
        violations = self.analizer.violation_analysis(model_process_res, methods)
        print(f'Найденные нарушения: {violations}')
        return violations
    
    def show_violations(self, timestamp, video_path):
        """метод для получения кадра со всеми метками использованных моделей обработки

        Args:
            timestamp (_type_): временная метка кадра
            video_path (_type_): путь к видео

        Returns:
            np.array: кадр со всеми метками использованных моделей
        """
        frame = self.renderer.get_frame(timestamp, video_path)
        
        frame_detections_data = self.handler.get_frame_data(timestamp, 'detection')
        frame = self.renderer.plot_boxes(frame, frame_detections_data, 'detection')

        frame_pos_data = self.handler.get_frame_data(timestamp, 'pos_est')
        frame = self.renderer.plot_boxes(frame, frame_pos_data, 'pos_est')
        return frame
