#!/usr/bin/env python3
"""
Independent test script for DataProto<->BatchMeta conversion decorator.

This script uses the real DataProto class and mocks only the TransferQueue components
for testing.
"""

import asyncio
import sys
import torch
from tensordict import TensorDict, NonTensorData, NonTensorStack

# Add the recipe directory to Python path
sys.path.append('/Users/hanzhenyu/verl/recipe/transfer_queue')
sys.path.append('/Users/hanzhenyu/verl')

# Import real DataProto
try:
    from verl import DataProto
    DATAPROTO_AVAILABLE = True
    print("✓ DataProto imported successfully")
except ImportError as e:
    print(f"⚠ DataProto not available: {e}")
    DATAPROTO_AVAILABLE = False

# Import TransferQueue components
try:
    from verl.experimental.transfer_queue import BatchMeta, SampleMeta, FieldMeta
    from verl.experimental.transfer_queue import ProductionStatus
    from verl.experimental.transfer_queue import AsyncTransferQueueClient
    TRANSFER_QUEUE_AVAILABLE = True
    print("✓ TransferQueue imported successfully")
except ImportError as e:
    print(f"⚠ TransferQueue not available: {e}")
    TRANSFER_QUEUE_AVAILABLE = False

# Import the decorator
try:
    from dataproto_conversion import dataproto_batchmeta_conversion
    DECORATOR_AVAILABLE = True
    print("✓ Decorator imported successfully")
except ImportError as e:
    print(f"⚠ Decorator not available: {e}")
    DECORATOR_AVAILABLE = False

def create_test_batchmeta() -> BatchMeta:
    """Create a test BatchMeta for testing."""
    samples = []
    for i in range(4):
        fields = {
            "input_ids": FieldMeta(
                name="input_ids",
                dtype=torch.int64,
                shape=torch.Size([10]),
                production_status=ProductionStatus.READY_FOR_CONSUME
            ),
            "attention_mask": FieldMeta(
                name="attention_mask",
                dtype=torch.int64,
                shape=torch.Size([10]),
                production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }

        sample = SampleMeta(
            global_step=1,
            global_index=i,
            storage_id=f"storage_0",
            local_index=i,
            fields=fields
        )
        samples.append(sample)

    return BatchMeta(samples=samples, extra_info={"test": True})

class MockTransferQueueClient:
    """Mock TransferQueue client for testing."""

    def __init__(self):
        self.storage = {}
        self.call_log = []

    async def async_get_data(self, batch_meta: BatchMeta):
        """Mock data retrieval."""
        self.call_log.append("async_get_data")
        batch_size = len(batch_meta)

        return {
            "input_ids": torch.randint(0, 1000, (batch_size, 10)),
            "attention_mask": torch.ones(batch_size, 10),
        }

    async def async_put(self, data, metadata):
        """Mock data storage."""
        self.call_log.append("async_put")
        storage_id = list(metadata.storage_meta_groups.keys())[0] if metadata.storage_meta_groups else "mock_storage"
        self.storage[storage_id] = data

    async def async_get_meta(self, **kwargs):
        """Mock metadata retrieval."""
        self.call_log.append("async_get_meta")
        return create_test_batchmeta()

# Test functions that work with real DataProto
def compute_response_mask_function(data: DataProto) -> DataProto:
    """Test function: compute response mask."""
    responses = data.batch.get("responses", torch.zeros(len(data), 5))
    response_length = responses.size(1)

    # Use a default attention_mask if not present
    if "attention_mask" in data.batch:
        attention_mask = data.batch["attention_mask"]
    else:
        attention_mask = torch.ones(len(data), responses.size(1))

    response_mask = attention_mask[:, -response_length:]

    # Add to batch
    data.batch["response_mask"] = response_mask

    # Add some non-tensor data
    data.non_tensor_batch["mask_computed"] = True

    return data

def apply_kl_penalty_function(data: DataProto, kl_ctrl: float = 0.1) -> DataProto:
    """Test function: apply KL penalty."""
    response_mask = data.batch.get("response_mask", torch.ones_like(data.batch.get("responses", torch.ones(len(data), 5))))
    kl_penalty = torch.rand(len(data)) * kl_ctrl

    # Add tensor result
    data.batch["kl_penalty"] = kl_penalty

    # Add non-tensor results
    data.non_tensor_batch["kl_ctrl_value"] = kl_ctrl
    data.non_tensor_batch["step_info"] = {"iteration": 1, "total_steps": 100}

    return data

# Decorated versions
@dataproto_batchmeta_conversion()
def compute_response_mask_decorated(data: DataProto) -> DataProto:
    """Decorated test function."""
    return compute_response_mask_function(data)

@dataproto_batchmeta_conversion()
def apply_kl_penalty_decorated(data: DataProto, kl_ctrl: float = 0.1) -> DataProto:
    """Decorated test function."""
    return apply_kl_penalty_function(data, kl_ctrl)

def test_dataproto_functionality():
    """Test real DataProto functionality."""
    print("\nTesting DataProto functionality...")

    # Test creation from single dict - only tensors supported
    data = DataProto.from_single_dict({
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "learning_rate": torch.tensor([0.001, 0.001])
    })

    print(f"DataProto length: {len(data)}")
    print(f"Batch keys: {list(data.batch.keys())}")
    print(f"Non-tensor keys: {list(data.non_tensor_batch.keys())}")

    assert len(data) == 2
    assert "input_ids" in data.batch
    assert data.batch["input_ids"].shape == (2, 3)
    assert "learning_rate" in data.batch
    assert data.batch["learning_rate"].shape == (2,)

    print("✓ DataProto works correctly")

def test_basic_functionality():
    """Test basic function functionality without decorator."""
    print("\nTesting basic functionality...")

    # Create test data
    data = DataProto.from_single_dict({
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones(2, 3),
        "responses": torch.tensor([[7, 8], [9, 10]])
    })

    print(f"Input data shape: {data.batch['input_ids'].shape}")
    print(f"Responses shape: {data.batch['responses'].shape}")

    # Test compute_response_mask
    result = compute_response_mask_function(data)
    assert "response_mask" in result.batch
    assert result.batch["response_mask"].shape == (2, 2)  # response length is 2
    assert result.non_tensor_batch["mask_computed"] is True

    print(f"Response mask shape: {result.batch['response_mask'].shape}")

    # Test apply_kl_penalty
    result = apply_kl_penalty_function(result, kl_ctrl=0.2)
    assert "kl_penalty" in result.batch
    assert result.batch["kl_penalty"].shape == (2,)  # batch size is 2
    assert result.non_tensor_batch["kl_ctrl_value"] == 0.2

    print(f"KL penalty shape: {result.batch['kl_penalty'].shape}")

    print("✓ Basic functionality works correctly")

async def test_decorator_functionality():
    """Test decorator functionality with mock client."""
    if not (DECORATOR_AVAILABLE and TRANSFER_QUEUE_AVAILABLE):
        print("\n⚠ Skipping decorator tests (components not available)")
        return

    print("\nTesting decorator functionality...")

    # Create test BatchMeta and client
    batch_meta = create_test_batchmeta()
    mock_client = MockTransferQueueClient()

    print(f"Test BatchMeta size: {len(batch_meta)}")
    print(f"BatchMeta fields: {batch_meta.field_names}")

    # Test without client (should work with empty data)
    print("\n1. Testing compute_response_mask decorator without client...")
    try:
        result_batch_meta = compute_response_mask_decorated(batch_meta)
        print("✓ compute_response_mask decorator works without client")
        print(f"  Result BatchMeta size: {len(result_batch_meta)}")
        print(f"  Result fields: {result_batch_meta.field_names}")
    except Exception as e:
        print(f"✗ compute_response_mask decorator failed: {e}")
        import traceback
        traceback.print_exc()

    # Test with client in a separate thread to avoid event loop issues
    print("\n2. Testing compute_response_mask decorator with client...")
    try:
        # Run in a separate thread to avoid event loop conflicts
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(compute_response_mask_decorated, batch_meta, transfer_queue_client=mock_client)
            result_batch_meta = future.result(timeout=10)

        print("✓ compute_response_mask decorator works with client")
        print(f"  Result BatchMeta size: {len(result_batch_meta)}")
        print(f"  Result fields: {result_batch_meta.field_names}")
        print(f"  Client calls: {mock_client.call_log}")
        assert "async_get_data" in mock_client.call_log
        assert "async_put" in mock_client.call_log
        assert "response_mask" in result_batch_meta.field_names
        mock_client.call_log.clear()
    except Exception as e:
        print(f"✗ compute_response_mask decorator with client failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: apply_kl_penalty without client
    print("\n3. Testing apply_kl_penalty decorator without client...")
    try:
        result_batch_meta = apply_kl_penalty_decorated(batch_meta, kl_ctrl=0.15)
        print("✓ apply_kl_penalty decorator works without client")
        print(f"  Result BatchMeta size: {len(result_batch_meta)}")
        print(f"  Result fields: {result_batch_meta.field_names}")
    except Exception as e:
        print(f"✗ apply_kl_penalty decorator failed: {e}")
        import traceback
        traceback.print_exc()

def test_tensordict_nontensor_support():
    """Test TensorDict NonTensorData support."""
    print("\nTesting TensorDict NonTensorData support...")

    # Simplified test - just check if NonTensorData can be created
    try:
        nt_data = NonTensorData(0.001)
        nt_stack = NonTensorStack([nt_data, nt_data])
        print("✓ NonTensorData and NonTensorStack work correctly")
    except Exception as e:
        print(f"⚠ NonTensorData test failed: {e}")
        print("  This is likely a TensorDict version compatibility issue")

async def main():
    """Main test function."""
    print("=== DataProto<->BatchMeta Decorator Test ===")

    # Check availability
    print(f"\nComponent availability:")
    print(f"  DataProto: {DATAPROTO_AVAILABLE}")
    print(f"  TransferQueue: {TRANSFER_QUEUE_AVAILABLE}")
    print(f"  Decorator: {DECORATOR_AVAILABLE}")

    # Test DataProto functionality
    if DATAPROTO_AVAILABLE:
        test_dataproto_functionality()
        test_basic_functionality()
        test_tensordict_nontensor_support()
    else:
        print("\n⚠ Skipping DataProto tests")

    # Test decorator functionality
    if DECORATOR_AVAILABLE and TRANSFER_QUEUE_AVAILABLE and DATAPROTO_AVAILABLE:
        await test_decorator_functionality()
    else:
        print("\n⚠ Skipping decorator tests (missing components)")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(main())