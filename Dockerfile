FROM continuumio/miniconda3

# copy our code into the container
WORKDIR /code
COPY . /code/

# set up conda: update, create env, activate env
RUN conda update -y -n base -c defaults conda
RUN conda create -y --name grids -c conda-forge --file requirements.txt
RUN echo "conda activate grids" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure uproot is installed:"
RUN python -c "import uproot"
