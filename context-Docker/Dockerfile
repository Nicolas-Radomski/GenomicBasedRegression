# install Python and requirements
FROM ashsmol/python312
# install system dependencies for lightgbm
RUN apt-get update && apt-get install -y libgomp1
# upgrade pip
RUN pip install --upgrade pip
# sets the working directory for any RUN, CMD, ENTRYPOINT, COPY and ADD instructions
WORKDIR /context-Docker
# copy the script and requirements
COPY GenomicBasedRegression.py ./
COPY requirements.txt ./
# install Python requirements
RUN pip install -r requirements.txt
# set entrypoint
ENTRYPOINT ["python3.12", "/context-Docker/GenomicBasedRegression.py"]
# optional: change working directory for runtime
WORKDIR /wd
