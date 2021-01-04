FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

WORKDIR /home/app

COPY . .

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libgtk2.0-dev
RUN apt-get install -y build-essential
RUN apt-get install -y manpages-dev

RUN pip install -r requirements.txt
RUN pip install pycocotools

# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
# CMD ["python", "app.py"]
RUN chmod +x /home/app/entrypoint.sh
CMD /home/app/entrypoint.sh
