FROM debian:latest
RUN apt-get update && apt-get install python3-pip -y && pip3 install fastapi
RUN pip install uvicorn
RUN pip install requests
RUN pip install matplotlib && pip install pandas && pip install seaborn && pip install sklearn && pip install imblearn
ADD git_exam_churn.py /home/ubuntu/exam_scoring/git_exam_churn.py
WORKDIR /home/ubuntu/exam_scoring/
EXPOSE 8000
CMD python3 git_exam_churn.py
