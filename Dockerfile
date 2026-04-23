FROM nvcr.io/nvidia/tensorflow:24.03-tf2-py3
RUN pip install --no-cache-dir opencv-python-headless xlwt
