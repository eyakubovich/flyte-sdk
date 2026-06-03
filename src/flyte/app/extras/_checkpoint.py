from __future__ import annotations

import os
import socket
import time

CHECKPOINT_REQUEST_HOST_ENV = "CHECKPOINT_REQUEST_HOST"
CHECKPOINT_REQUEST_PORT_ENV = "CHECKPOINT_REQUEST_PORT"
# Created by the app to request a checkpoint. Suffix is the process PID.
CHECKPOINT_POD_NAME_ENV = "CHECKPOINT_POD_NAME"

# Created by the runtime when restore is complete.
CHECKPOINT_RESTORED_FILENAME_ENV = "CHECKPOINT_RESTORED_FILENAME"

def checkpoint() -> None:
    req_host = os.environ.get(CHECKPOINT_REQUEST_HOST_ENV)
    req_port = os.environ.get(CHECKPOINT_REQUEST_PORT_ENV, "7321")
    pod_name = os.environ.get(CHECKPOINT_POD_NAME_ENV)
    restore_path = os.environ.get(CHECKPOINT_RESTORED_FILENAME_ENV)

    if not req_host:
        print(f"{CHECKPOINT_REQUEST_HOST_ENV} is not set, skipping checkpoint")
        return

    if not pod_name:
        print(f"{CHECKPOINT_POD_NAME_ENV} is not set, skipping checkpoint")
        return

    # The env vars don't change on restore so they all have to be set during checkpointing.
    if not restore_path:
        print(f"{CHECKPOINT_RESTORED_FILENAME_ENV} is not set, skipping checkpoint")
        return

    print(f"Request addr: {req_host}:{req_port}")

    # Create the request file to signal the runtime to checkpoint.
    checkpoint_request(req_host, int(req_port), pod_name)

    # The checkpoint restore will put us here-ish again.
    # During checkpointing, the restore file will not exist and we block here
    # until the checkpoint is complete and the pod is killed.
    # During restore, the runtime creates the restore file and inotify unblocks us.

    _wait_for_file(restore_path)

def checkpoint_request(host: str, port: int, pod_name: str) -> None:
    sock = socket.socket()
    sock.connect((host, port))
    sock.sendall(f"{pod_name}\n".encode())
    sock.close()

def _wait_for_file(path: str) -> None:
    # It is suprisingly difficult to do this without polling.
    # inotify/fanotify don't work because the file is created prior to restore.
    while not os.path.exists(path):
        print(f"Waiting for file: {path}")
        time.sleep(0.1)

    print(f"File found: {path}")
