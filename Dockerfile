# Definition of Submission container
ARG DOCKER_REGISTRY=docker.io
ARG ARCH=amd64
ARG MAJOR=daffy
ARG BASE_TAG=${MAJOR}-${ARCH}

FROM light5551/fast-build:daffy-amd64
#${DOCKER_REGISTRY}/duckietown/dt-machine-learning-base-environment:${BASE_TAG}

#ARG PIP_INDEX_URL="https://pypi.org/simple"
#ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# Setup any additional pip packages
#COPY requirements.* ./
#RUN cat requirements.* > .requirements.txt
#RUN apt-get install freeglut3-dev
#RUN python3 -m pip install --no-cache-dir -r .requirements.txt
RUN pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu117 -U
#RUN pip3 uninstall dataclasses -y
# let's copy all our solution files to our workspace

WORKDIR /submission
COPY lane_control /submission/lane_control
COPY env /submission/env
COPY learning /submission/learning
COPY solution.py ./
COPY ml /submission/ml
COPY ml_model /submission/ml_model

CMD ["python3", "./solution.py"]