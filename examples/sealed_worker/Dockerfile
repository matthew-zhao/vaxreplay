ARG BASE_IMAGE=python:3.12-alpine
FROM ${BASE_IMAGE}

COPY worker.py /opt/vaxreplay/worker.py
COPY isolation_probe.py /opt/vaxreplay/isolation_probe.py

# VaxReplay overrides the image entrypoint with the manifest's complete argv. CMD and VOLUME must
# remain empty because the runner rejects implicit arguments and writable image-declared mounts.
ENTRYPOINT ["/usr/local/bin/python", "/opt/vaxreplay/worker.py"]
CMD []
