'''
    Файл конфигурации системы. Содержит информацию об используемых моделях обработки, 
    параметрах обработки видео и о доступных методов анализа.
'''

config = {
    'handler':{
        'models': {
            'yolo_phone_detection_heavy': {
                'path': './trained_models/70_7_x_best.pt',
                'format': 'YOLO',
                'task': 'detection',
                'classes' : None,
                'specific_params': {
                    'conf': 0.8
                }
            },
            'yolo_phone_detection_light': {
                'path': './trained_models/yolov8m.pt', 
                'format': 'YOLO',
                'task': 'detection',
                'classes': ['67'],
                'specific_params': {
                    'conf': 0.7
                }
            },
            'yolo_pose_detection_heavy': {
                'path': './trained_models/yolov8x-pose.pt',
                'format': 'YOLO',
                'task': 'pos_est',
                'specific_params': {
                    'conf': 0.8
                }
            },
            'yolo_pose_detection_light': {
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
            'yolo_phone_detection_light',
            'yolo_pose_detection_light'
        ]
    },
    'analyser': {
        'methods': {
            'wrist_phone_usage': {
                'required_data': ['detection', 'pos_est'],
                'min_duration': 3000,
                'max_wrist_dist': 200,
                'max_short_diff': 3000,
                'max_long_diff': 5000
            }
        },
        'default_methods': [
            'wrist_phone_usage',
        ]
    }
}