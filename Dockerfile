FROM centos:latest
RUN yum install openssh-server -y
RUN yum install python36 -y
RUN yum install net-tools -y
RUN pip3 install numpy
RUN pip3 install keras
RUN pip3 install scikit-learn
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install seaborn
RUN pip3 install --upgrade pip
ENTRYPOINT ["python3"]

