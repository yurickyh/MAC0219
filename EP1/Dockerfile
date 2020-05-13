FROM debian:stable-slim
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y --no-install-recommends linux-perf && \
    ln -s /usr/bin/perf_* /usr/bin/perf && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
