'''
    Файл конфигурации системы. Содержит информацию об используемых моделях обработки, 
    параметрах обработки видео и о доступных методов анализа.
'''

config = {
    'handler':{
        'models': {
            'модель детекции (тяжелая)': {
                'path': './trained_models/70_7_x_best.pt',
                'format': 'YOLO',
                'task': 'detection',
                'specific_params': {
                    'conf': 0.8,
                    'classes' : [0]
                }
            },
            'модель детекции (легкая)': {
                'path': './trained_models/yolov8m.pt', 
                'format': 'YOLO',
                'task': 'detection',
                'specific_params': {
                    'conf': 0.7,
                    'classes': [67]
                }
            },
            'модель определеня позы (тяжелая)': {
                'path': './trained_models/yolov8x-pose.pt',
                'format': 'YOLO',
                'task': 'pos_est',
                'specific_params': {
                    'conf': 0.8
                }
            },
            'модель определеня позы (легкая)': {
                'path': './trained_models/yolov8m-pose.pt',
                'format': 'YOLO',
                'task': 'pos_est',
                'specific_params': {
                    'conf': 0.8
                }
            },
        },
        'processing': {
            'BATCH_SIZE': 4,
            'save_path': None
        },
        'default_models': [
            'модель детекции (тяжелая)',
            'модель определеня позы (тяжелая)'
        ]
    },
    'analyser': {
        'methods': {
            'wrist_usage': {
                'required_data': ['detection', 'pos_est'],
                'min_duration': 3000,
                'max_wrist_dist': 200,
                'max_short_diff': 3000,
                'max_long_diff': 5000,
                'detected_classes': ['cell phone', 'cell phones']
            }
        },
        'default_methods': [
            'wrist_usage',
        ]
    }
}