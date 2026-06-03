import json
import time
import urllib.request
import logging
import os
import threading
from typing import Generator, Optional

import torch
from flyte.app.extras._model_loader.config import (
    LOCAL_MODEL_PATH,
    REMOTE_MODEL_PATH,
    STREAM_SAFETENSORS,
)
from flyte.app.extras._model_loader.loader import SafeTensorsStreamer, prefetch
from flyte.app.extras import checkpoint
from flyteplugins.vllm._constants import VLLM_MIN_VERSION, VLLM_MIN_VERSION_STR

try:
    import vllm
except ImportError:
    raise ImportError(f"vllm is not installed. Please install 'vllm>={VLLM_MIN_VERSION_STR}', to use the model loader.")

if tuple([int(part) for part in vllm.__version__.split(".") if part.isdigit()]) < VLLM_MIN_VERSION:
    raise ImportError(
        f"vllm version >={VLLM_MIN_VERSION_STR} required, but found {vllm.__version__}. Please upgrade vllm."
    )

import vllm.entrypoints.cli.main
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader

try:
    from vllm.model_executor.model_loader.utils import set_default_torch_dtype
except ImportError:
    # vllm 0.13.0 moved the set_default_torch_dtype to vllm.utils.torch_utils
    from vllm.utils.torch_utils import set_default_torch_dtype

logger = logging.getLogger(__name__)


class FlyteModelLoader(DefaultModelLoader):
    """Custom model loader for streaming model weights from object storage."""

    def _get_weights_iterator(
        self, source: DefaultModelLoader.Source
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        # Try to load weights using the Flyte SafeTensorsLoader. Fallback to the default loader otherwise.
        try:
            streamer = SafeTensorsStreamer(REMOTE_MODEL_PATH, LOCAL_MODEL_PATH)
        except ValueError:
            yield from super()._get_weights_iterator(source)
        else:
            for name, tensor in streamer.get_tensors():
                yield source.prefix + name, tensor

    def download_model(self, model_config: ModelConfig) -> None:
        # This model loader supports streaming only
        pass

    def _load_sharded_model(self, vllm_config: VllmConfig, model_config: ModelConfig) -> torch.nn.Module:
        # Forked from: https://github.com/vllm-project/vllm/blob/99d01a5e3d5278284bad359ac8b87ee7a551afda/vllm/model_executor/model_loader/loader.py#L613
        # Sanity checks
        tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
        rank = get_tensor_model_parallel_rank()
        if rank >= tensor_parallel_size:
            raise ValueError(f"Invalid rank {rank} for tensor parallel size {tensor_parallel_size}")
        with set_default_torch_dtype(vllm_config.model_config.dtype):  # type: ignore[arg-type]
            with torch.device(vllm_config.device_config.device):  # type: ignore[arg-type]
                model_loader = DummyModelLoader(load_config=vllm_config.load_config)
                model = model_loader.load_model(vllm_config=vllm_config, model_config=model_config)
                for i, (name, module) in enumerate(model.named_modules()):
                    print(i, name, module)
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)
            state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
            streamer = SafeTensorsStreamer(
                REMOTE_MODEL_PATH,
                LOCAL_MODEL_PATH,
                rank=rank,
                tensor_parallel_size=tensor_parallel_size,
            )
            for name, tensor in streamer.get_tensors():
                # If loading with LoRA enabled, additional padding may
                # be added to certain parameters. We only load into a
                # narrowed view of the parameter data.
                param_data = state_dict[name].data
                param_shape = state_dict[name].shape
                for dim, size in enumerate(tensor.shape):
                    if size < param_shape[dim]:
                        param_data = param_data.narrow(dim, 0, size)
                if tensor.shape != param_shape:
                    logger.warning(
                        "loading tensor of shape %s into parameter '%s' of shape %s",
                        tensor.shape,
                        name,
                        param_shape,
                    )
                param_data.copy_(tensor)
                state_dict.pop(name)
            if state_dict:
                raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")
        return model.eval()

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
    ) -> torch.nn.Module:
        logger.info("Loading model with FlyteModelLoader")
        if vllm_config.parallel_config.tensor_parallel_size > 1:
            return self._load_sharded_model(vllm_config, model_config)
        else:
            return super().load_model(vllm_config, model_config)


if REMOTE_MODEL_PATH and STREAM_SAFETENSORS:
    register_model_loader("flyte-vllm-streaming")(FlyteModelLoader)


async def _get_model_files():
    import flyte.storage as storage

    if not await storage.exists(REMOTE_MODEL_PATH):
        raise FileNotFoundError(f"Model path not found: {REMOTE_MODEL_PATH}")

    await prefetch(
        REMOTE_MODEL_PATH,
        LOCAL_MODEL_PATH,
        exclude_safetensors=STREAM_SAFETENSORS,
    )

class VLLMModelCheckpoint:
    port: str
    model_id: str

    def __init__(self, port: str, model_id: str):
        self.port = port
        self.model_id = model_id

    def warm_up(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": "Hello, LLM!"}],
            "max_completion_tokens": 16,
        }

        success = 0
        while success < 3:
            try:
                self._post(url, payload)
                success += 1
            except Exception as e:
                print(f"Failed to warm up: {e}, retrying...")
                time.sleep(1)

    def sleep(self):
        url = f"http://localhost:{self.port}/sleep?level=1"
        self._post(url)

    def wakeup(self):
        url = f"http://localhost:{self.port}/wake_up"
        self._post(url)

    def _post(self, url: str, payload: Optional[dict[str, str]] = None) -> None:
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        else:
            data = None
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        resp = urllib.request.urlopen(req)
        if resp.status >= 400:
            raise Exception(f"Failed to post: {resp.status}")

    def pre_checkpoint(self):
        self.warm_up()
        self.sleep()

    def post_restore(self):
        self.wakeup()

def do_checkpoint():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser("serve", help="Serve the model")
    serve_parser.add_argument("--port", type=str, required=True)
    serve_parser.add_argument("--served-model-name", type=str, required=True)
    args = parser.parse_known_args()[0]
    assert args.command == "serve"

    cp = VLLMModelCheckpoint(args.port, args.served_model_name)
    cp.pre_checkpoint()
    checkpoint()
    cp.post_restore()

def main():
    import asyncio

    # TODO: add CLI here to be able to pass in serialized parameters from AppEnvironment
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("REMOTE_MODEL_PATH: %s", REMOTE_MODEL_PATH)
    logger.info("LOCAL_MODEL_PATH: %s", LOCAL_MODEL_PATH)
    logger.info("STREAM_SAFETENSORS: %s", STREAM_SAFETENSORS)

    if REMOTE_MODEL_PATH:
        logger.info("Prefetching model files from object storage...")
        asyncio.run(_get_model_files())

    if os.getenv("FLYTE_CHECKPOINT_SIGNAL_DIR"):
        threading.Thread(target=do_checkpoint).start()

    vllm.entrypoints.cli.main.main()
