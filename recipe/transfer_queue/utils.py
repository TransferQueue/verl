# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Huawei Ltd. and/or its affiliates
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

import asyncio
import torch
import numpy as np

from tensordict import TensorDict, NonTensorStack

from verl.experimental.transfer_queue.client import TransferQueueClient
from verl.experimental.transfer_queue.metadata import BatchMeta
from verl.protocol import DataProto

def _tensordict_to_dataproto(td: TensorDict, extra_info: dict) -> DataProto:
    """Convert a tensordict.TensorDict to a DataProto.

    Args:
        td (tensordict.TensorDict): The input TensorDict.
        extra_info (dict): Additional metadata to include in the DataProto.

    Returns:
        DataProto: The converted DataProto.
    """
    batch = td.select(*[k for k, v in td.items() if isinstance(v, torch.Tensor)])
    non_tensor_batch = {}
    for k, v in td.items():
        if not isinstance(v, torch.Tensor):
            if isinstance(v, NonTensorStack):
                v = v.tolist()
                for item in v:
                    if not isinstance(item, np.ndarray):
                        raise ValueError("DataProto only supports tensor or NonTensorStack in tensordict")
                v = np.stack(v)
                non_tensor_batch[k] = v
            if isinstance(v, NonTensorData):
                v = v.data
                assert isinstance(v, np.ndarray), "DataProto only supports tensor or NonTensorStack in tensordict"
                non_tensor_batch[k] = v
            else:
                raise ValueError("DataProto only supports tensor or NonTensorStack in tensordict")
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=extra_info)

def batchmeta_to_dataproto(
        client: TransferQueueClient, 
        batch_meta: BatchMeta,
    ) -> DataProto:
    """Convert BatchMeta to DataProto by fetching the actual data from the TransferQueueClient.

    Args:
        client (TransferQueueClient): The client to fetch data from.
        batch_meta (BatchMeta): The metadata of the batch containing references to the actual data.

    Returns:
        DataProto: The complete data batch with tensors and non-tensor data.
    """
    data = asyncio.run(client.async_get_data(batch_meta))
    return _tensordict_to_dataproto(data, batch_meta.extra_info)


# test _tensordict_to_dataproto
if __name__ == "__main__":
    from tensordict import TensorDict, NonTensorData
    output = torch.randn(2, 8)
    td = TensorDict(
        {
            "generate_sequences_ids": output,
            "non_tensor_data": NonTensorData(np.array([[1, 2, 3], [1, 2, 3]])),
            "nested_tensor": torch.nested.as_nested_tensor([torch.randn(1, 2) for _ in range(output.size(0))]),
        },
        batch_size=output.size(0),
    )
    dp = _tensordict_to_dataproto(td, {"extra_key": "extra_value"})



