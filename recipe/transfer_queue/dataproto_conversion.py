# Copyright 2025 The TransferQueue Team.
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

"""
DataProto conversion decorator for TransferQueue integration.

This decorator wraps functions that take DataProto as input and return DataProto as output,
enabling them to work with BatchMeta and TransferQueue system.

Pattern:
1. Input: BatchMeta + TransferQueueClient
2. Decorator: BatchMeta -> DataProto -> function(DataProto) -> DataProto -> update BatchMeta
3. Output: Updated BatchMeta
"""

import asyncio
import functools
import logging
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict, NonTensorData, NonTensorStack

from verl import DataProto
from verl.experimental.transfer_queue import AsyncTransferQueueClient, BatchMeta

logger = logging.getLogger(__name__)


def dataproto_batchmeta_conversion(transfer_queue_client: Optional[AsyncTransferQueueClient] = None):
    """
    Decorator for converting DataProto functions to work with BatchMeta.

    This decorator enables DataProto-based functions to work with TransferQueue's
    BatchMeta system by:
    1. Converting BatchMeta input to DataProto via client
    2. Calling the wrapped function with DataProto
    3. Converting function's DataProto output back to update BatchMeta
    4. Returning the updated BatchMeta

    Args:
        transfer_queue_client: AsyncTransferQueueClient for data operations

    Usage:
        @dataproto_batchmeta_conversion(client)
        def apply_kl_penalty(data: DataProto, kl_ctrl) -> DataProto:
            # Function works with DataProto as usual
            response_mask = data.batch["response_mask"]
            # ... compute kl_penalty ...
            data.batch["kl_penalty"] = kl_penalty_result
            return data

        # Usage with BatchMeta:
        batch_meta = apply_kl_penalty(batch_meta, kl_ctrl, transfer_queue_client=client)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract batch_meta and client from arguments
            batch_meta, client, other_args, other_kwargs = _extract_args(args, kwargs, transfer_queue_client)

            # Convert BatchMeta to DataProto
            data = await _batchmeta_to_dataproto_async(batch_meta, client)

            # Call function with DataProto
            result_data = await func(data, *other_args, **other_kwargs)

            # Update BatchMeta with result
            await _update_batchmeta_with_result_async(result_data, batch_meta, client)

            return batch_meta

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Extract batch_meta and client from arguments
            batch_meta, client, other_args, other_kwargs = _extract_args(args, kwargs, transfer_queue_client)

            # Convert BatchMeta to DataProto
            data = _batchmeta_to_dataproto_sync(batch_meta, client)

            # Call function with DataProto
            result_data = func(data, *other_args, **other_kwargs)

            # Update BatchMeta with result
            _update_batchmeta_with_result_sync(result_data, batch_meta, client)

            return batch_meta

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def _extract_args(args: tuple, kwargs: dict, default_client: Optional[AsyncTransferQueueClient]):
    """Extract batch_meta, client, and other arguments from function call."""
    # Find batch_meta (first argument)
    batch_meta = args[0] if args else None

    # Find client in kwargs or use default
    client = kwargs.pop('transfer_queue_client', default_client)

    # Remaining arguments
    other_args = args[1:] if len(args) > 1 else ()
    other_kwargs = kwargs

    return batch_meta, client, other_args, other_kwargs


def _batchmeta_to_dataproto_sync(batch_meta: BatchMeta, client: Optional[AsyncTransferQueueClient]) -> DataProto:
    """Convert BatchMeta to DataProto (synchronous)."""
    if client is not None:
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in a running loop, this shouldn't happen for sync wrapper
            raise RuntimeError("Sync wrapper called from within async context")
        except RuntimeError:
            # No running loop, we can use asyncio.run
            data_dict = asyncio.run(client.async_get_data(batch_meta))
    else:
        # For testing without client, return empty DataProto
        data_dict = {}

    return _dict_to_dataproto(data_dict, batch_meta.extra_info)


async def _batchmeta_to_dataproto_async(batch_meta: BatchMeta, client: Optional[AsyncTransferQueueClient]) -> DataProto:
    """Convert BatchMeta to DataProto (asynchronous)."""
    if client is not None:
        # Get data from storage
        data_dict = await client.async_get_data(batch_meta)
    else:
        # For testing without client, return empty DataProto
        data_dict = {}

    return _dict_to_dataproto(data_dict, batch_meta.extra_info)


def _update_batchmeta_with_result_sync(result_data: DataProto, batch_meta: BatchMeta, client: Optional[AsyncTransferQueueClient]):
    """Update BatchMeta with DataProto result (synchronous)."""
    # Convert DataProto to TensorDict
    output_tensor_dict = _dataproto_to_tensordict(result_data)

    if client is not None:
        # Store output data
        asyncio.run(client.async_put(data=output_tensor_dict, metadata=batch_meta))

    # Update BatchMeta with new fields
    batch_meta.add_fields(output_tensor_dict)


async def _update_batchmeta_with_result_async(result_data: DataProto, batch_meta: BatchMeta, client: Optional[AsyncTransferQueueClient]):
    """Update BatchMeta with DataProto result (asynchronous)."""
    # Convert DataProto to TensorDict
    output_tensor_dict = _dataproto_to_tensordict(result_data)

    if client is not None:
        # Store output data
        await client.async_put(data=output_tensor_dict, metadata=batch_meta)

    # Update BatchMeta with new fields
    batch_meta.add_fields(output_tensor_dict)


def _dict_to_dataproto(data_dict: dict, meta_info: dict) -> DataProto:
    """Convert dictionary to DataProto, handling NonTensorData."""
    batch = {}
    non_tensor_batch = {}

    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value
        elif isinstance(value, NonTensorStack):
            # Convert NonTensorStack back to list format for DataProto
            non_tensor_batch[key] = [item.data for item in value]
        elif isinstance(value, NonTensorData):
            # Convert NonTensorData back to scalar
            non_tensor_batch[key] = value.data
        else:
            # Keep other types as-is
            non_tensor_batch[key] = value

    # Determine batch size
    batch_size = len(next(iter(batch.values()), [])) if batch else 0

    return DataProto(
        batch=TensorDict(batch, batch_size=batch_size),
        non_tensor_batch=non_tensor_batch,
        meta_info=meta_info.copy()
    )


def _dataproto_to_tensordict(data: DataProto) -> TensorDict:
    """Convert DataProto to TensorDict for storage using NonTensorData."""
    # Start with tensor data
    tensor_dict = dict(data.batch)

    # Handle non-tensor data using NonTensorData/NonTensorStack
    non_tensor_dict = {}
    for key, value in data.non_tensor_batch.items():
        if isinstance(value, torch.Tensor):
            # Keep tensors as-is
            tensor_dict[key] = value
        elif isinstance(value, (list, tuple)) and len(value) == len(data):
            # Batch-aligned lists: convert to NonTensorStack
            non_tensor_elements = []
            for item in value:
                if isinstance(item, (int, float, bool, str)):
                    non_tensor_elements.append(NonTensorData(item))
                else:
                    # For complex objects, keep as-is and let NonTensorData handle
                    non_tensor_elements.append(NonTensorData(item))
            non_tensor_dict[key] = NonTensorStack(non_tensor_elements)
        elif isinstance(value, (int, float, bool, str)):
            # Scalar values: broadcast to all samples using NonTensorData
            scalar_data = NonTensorData(value)
            non_tensor_dict[key] = NonTensorStack([scalar_data] * len(data))
        else:
            # Other types: try to preserve as NonTensorData
            try:
                scalar_data = NonTensorData(value)
                non_tensor_dict[key] = NonTensorStack([scalar_data] * len(data))
            except Exception:
                logger.warning(f"Could not convert non-tensor field {key} to NonTensorData, skipping")

    # Create TensorDict with non-tensor data - simplified approach
    try:
        if non_tensor_dict:
            return TensorDict(
                source=tensor_dict,
                batch_size=len(data),
                non_tensor_data=non_tensor_dict
            )
        else:
            return TensorDict(
                source=tensor_dict,
                batch_size=len(data)
            )
    except Exception as e:
        # Fallback: create empty TensorDict and add keys one by one
        logger.warning(f"TensorDict creation failed: {e}, using fallback method")
        if len(data) == 0:
            # Handle empty case
            td = TensorDict({}, batch_size=1)
            for key, value in tensor_dict.items():
                td.set(key, value)
            td.batch_size = len(data)  # Fix batch size
        else:
            td = TensorDict({}, batch_size=len(data))
            for key, value in tensor_dict.items():
                td.set(key, value)

        if non_tensor_dict:
            td.non_tensor_data = non_tensor_dict

        return td


def dataproto_batchmeta_conversion_v2(func: Optional[Callable] = None, *, transfer_queue_client: Optional[AsyncTransferQueueClient] = None):
    """
    Alternative decorator syntax that supports both @decorator and @decorator() usage.
    """
    def decorator(f: Callable) -> Callable:
        return dataproto_batchmeta_conversion(transfer_queue_client)(f)

    if func is not None:
        return decorator(func)
    return decorator