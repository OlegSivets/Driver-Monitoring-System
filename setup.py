from setuptools import setup, find_packages
import os.path as osp

def get_requirements(filename='requirements.txt'):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires

setup(
   name='dms',
   version='1.0',
   description='''
               Модуль для обработки видео с помощью различных инструментов с целью выявления нарушений правил безопасности
               Module for video processing with various tools to detect security violations''',
   author='Oleg Sivets',
   author_email='sivets-oleg-d@mail.ru',
   packages=find_packages(), 
   install_requires=get_requirements(),
   keywords=['Deep Learning', 'Computer Vision', 'Driwer Monitoring System'],
)