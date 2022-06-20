FROM debian:latest
RUN apt-get update && apt-get install python3-pip -y && pip3 install fastapi
RUN pip install uvicorn
RUN pip install requests
RUN pip install matplotlib && pip install pandas && pip install seaborn && pip install sklearn && pip install imblearn
ADD server_Fraude_detection.py /home/ubuntu/exam_scoring/git_exam_churnserver_Fraude_detection.py
WORKDIR /home/ubuntu/exam_scoring/git_exam_churn
EXPOSE 8000
CMD python3 server_Fraude_detection.py
