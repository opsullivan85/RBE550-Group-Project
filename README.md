# README

## Running our project

### Docker setup

Be sure you have docker installed locally on your computer. Then you can start it as shown below, or by using the devcontainer functionality of VSCode.

``` shell
cd .devcontainer

# start up the container
docker compose up --detach

# open a shell in the container
docker exec -it rbe-550-sim-1 bash

# navigate to the directory mapped to the project directory
cd ws

# do things in the container...

# stop the container once done
docker compose stop
```

Once the docker container is running, you can build the C++ component of our project. The instructions for this can be found [here](./src/quadruped_drake/README.md).

With the C++ part built, you can run the python program

```shell
cd src

python3 dogmeetsworld.py
```

At this point you can open `localhost:7000` in a webbrowser to see the environment visualization.

