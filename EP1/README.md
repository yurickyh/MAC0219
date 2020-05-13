# EP01 - MAC0219

This project is using Docker and we recommend using containers to run it locally.

## Build

To build a Docker Image for the project just execute the following command at `/EP1`:

```bash
docker image build -t ep1_paralela .
```

This should create a new Docker image in your local machine. You can look for it using:

```bash
docker image ls
```

## Runnning

Once you've built a image for the project, use the following command to run it:

```bash
docker container run --rm --security-opt seccomp=seccomp.json -v $PWD:/usr/src -it ep1_paralela:latest bash
```

There you go! This command will start a Docker container using the image built in the previous step and initialize a `bash` session.
