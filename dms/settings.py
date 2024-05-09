'''
    Файл конфигурации системы. Содержит информацию об используемых моделях обработки, 
    параметры обработки видео и параметры доступных методов анализа.
'''

config = {
    'handler':{
        'models': {
            'yolo_phone_detection': {
                'path': './trained_models/70_7_x_best.pt',
                'format': 'YOLO',
                'task': 'detection',
                'specific_params': {
                    'conf': 0.8
                }
            },
            'yolo_pose_detection': {
                'path': './trained_models/yolov8x-pose.pt',
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
        }
    },
    'analyser': {
        'wrist_phone_usage': {
            'required_data': ['detection', 'pos_est'],
            'min_duration': 3000,
            'max_wrist_dist': 200,
            'max_short_diff': 2000,
            'max_long_diff': 5000
        }
    }
}