cd /home/ubuntu/exam_scoring/git_exam_churn


docker kill $(docker ps -q)

docker image rm image_exam_scoring

docker image build . -t image_exam_scoring:latest

docker-compose up
