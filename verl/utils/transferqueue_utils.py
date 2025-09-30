# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from functools import wraps
from typing import Any

from transfer_queue import BatchMeta

from verl.experimental.transfer_queue import ZMQServerInfo

_TRANSFER_QUEUE_CONTROLLER_INFOS = None
_TRANSFER_QUEUE_STORAGE_INFOS = None


def set_transferqueue_server_info(controller_infos: dict[Any, ZMQServerInfo], storage_infos: dict[Any, ZMQServerInfo]):
    global _TRANSFER_QUEUE_CONTROLLER_INFOS, _TRANSFER_QUEUE_STORAGE_INFOS
    if _TRANSFER_QUEUE_CONTROLLER_INFOS is not None and _TRANSFER_QUEUE_STORAGE_INFOS is not None:
        return
    _TRANSFER_QUEUE_CONTROLLER_INFOS = controller_infos
    _TRANSFER_QUEUE_STORAGE_INFOS = storage_infos


def get_transferqueue_server_info():
    assert _TRANSFER_QUEUE_CONTROLLER_INFOS is not None and _TRANSFER_QUEUE_STORAGE_INFOS is not None, (
        "TransferQueue server infos have not been set yet."
    )
    return _TRANSFER_QUEUE_CONTROLLER_INFOS, _TRANSFER_QUEUE_STORAGE_INFOS


def _find_batchmeta(*args, **kwargs):
    for arg in args:
        if isinstance(arg, BatchMeta):
            return arg
    for v in kwargs.values():
        if isinstance(v, BatchMeta):
            return v
    return None


def _batchmeta_to_dataproto(batchmeta: BatchMeta):
    ...


def _update_batchmeta_with_output(output, batchmeta: BatchMeta):
    ...


async def _async_update_batchmeta_with_output(output, batchmeta: BatchMeta):
    ...


def batchmeta_dataproto_pipe():
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            batchmeta = _find_batchmeta(*args, **kwargs)
            if batchmeta is None:
                return func(*args, **kwargs)
            else:
                args = [_batchmeta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
                kwargs = {k: _batchmeta_to_dataproto(v) if isinstance(v, BatchMeta) else v for k, v in kwargs.items()}
                output = func(*args, **kwargs)
                _update_batchmeta_with_output(output, batchmeta)
                return batchmeta
            
        @wraps(func)
        async def async_inner(*args, **kwargs):
            batchmeta = _find_batchmeta(*args, **kwargs)
            if batchmeta is None:
                return await func(*args, **kwargs)
            else:
                args = [_batchmeta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
                kwargs = {k: _batchmeta_to_dataproto(v) if isinstance(v, BatchMeta) else v for k, v in kwargs.items()}
                output = await func(*args, **kwargs)
                await _async_update_batchmeta_with_output(output, batchmeta)
                return batchmeta

        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        return wrapper
    return decorator

