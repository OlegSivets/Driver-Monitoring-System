FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

COPY . /usr/dms

WORKDIR /usr/dms

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

ENTRYPOINT [ "python", "-m", "dms.interface.interface" ]
