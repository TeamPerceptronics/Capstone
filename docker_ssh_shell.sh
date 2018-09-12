#!/bin/bash
# opens ssh shell on running docker
# $1 is the id of the running docker instance
docker exec -it $1 /bin/bash
