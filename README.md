# README
# Running a Python Script in a Docker Container

``` shell
cd .devcontainer

# start up the container
docker compose up --detach

# open a shell in the container
docker exec -it rbe-550-sim-1

# navigate to the directory mapped to the project directory
cd ws

# run the example environment generation script
python3 example-gen-environment
```

At this point you can open `localhost:7000` in a webbrowser to see the environment visualization.

``` shell
# exit the script with Ctrl-C
# exit the container with Ctrl-D

# stop the container
docker compose stop
```

