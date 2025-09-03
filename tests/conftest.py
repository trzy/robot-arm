import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np
from typing import Dict, Any, Generator


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Create a temporary file in the temp directory."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("test content")
    return temp_file_path


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "robot": {
            "port": "/dev/ttyUSB0",
            "baudrate": 1000000,
            "servo_ids": [1, 2, 3, 4, 5, 6, 7]
        },
        "camera": {
            "width": 640,
            "height": 480,
            "fps": 30
        },
        "network": {
            "host": "localhost",
            "tcp_port": 5000,
            "udp_port": 5001
        },
        "dataset": {
            "output_dir": "/tmp/datasets",
            "compression": "gzip"
        }
    }


@pytest.fixture
def mock_serial_connection():
    """Mock serial connection for robot communication tests."""
    mock_serial = Mock()
    mock_serial.is_open = True
    mock_serial.read.return_value = b'\x00\x01\x02'
    mock_serial.write.return_value = 3
    mock_serial.in_waiting = 0
    return mock_serial


@pytest.fixture
def mock_dynamixel_handler():
    """Mock Dynamixel SDK handler for robot tests."""
    mock_handler = Mock()
    mock_handler.openPort.return_value = True
    mock_handler.setBaudRate.return_value = True
    mock_handler.ping.return_value = 0
    mock_handler.read2ByteTxRx.return_value = (1024, 0)
    mock_handler.write2ByteTxRx.return_value = 0
    return mock_handler


@pytest.fixture
def mock_camera():
    """Mock camera for vision tests."""
    mock_cam = Mock()
    
    # Create a sample image (480x640x3 RGB)
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mock_cam.read.return_value = (True, sample_image)
    mock_cam.isOpened.return_value = True
    mock_cam.release.return_value = None
    
    return mock_cam


@pytest.fixture
def sample_trajectory_data():
    """Sample trajectory data for ACT model testing."""
    return {
        "observations": np.random.randn(100, 14),  # 100 timesteps, 14 dims
        "actions": np.random.randn(100, 7),        # 100 timesteps, 7 joint actions
        "rewards": np.random.randn(100),           # 100 timesteps
        "dones": np.zeros(100, dtype=bool)         # 100 timesteps
    }


@pytest.fixture
def mock_tcp_socket():
    """Mock TCP socket for networking tests."""
    mock_socket = Mock()
    mock_socket.bind.return_value = None
    mock_socket.listen.return_value = None
    mock_socket.accept.return_value = (Mock(), ("127.0.0.1", 12345))
    mock_socket.recv.return_value = b'{"type": "test", "data": {}}'
    mock_socket.send.return_value = 32
    mock_socket.close.return_value = None
    return mock_socket


@pytest.fixture
def mock_udp_socket():
    """Mock UDP socket for networking tests."""
    mock_socket = Mock()
    mock_socket.bind.return_value = None
    mock_socket.recvfrom.return_value = (b'test_data', ("127.0.0.1", 12346))
    mock_socket.sendto.return_value = 9
    mock_socket.close.return_value = None
    return mock_socket


@pytest.fixture
def sample_hdf5_dataset(temp_dir: Path):
    """Create a sample HDF5 dataset file for testing."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not available for HDF5 dataset fixture")
    
    dataset_path = temp_dir / "sample_dataset.hdf5"
    
    with h5py.File(dataset_path, 'w') as f:
        # Create sample episode data
        episode_grp = f.create_group('episode_0')
        episode_grp.create_dataset('observations', data=np.random.randn(50, 14))
        episode_grp.create_dataset('actions', data=np.random.randn(50, 7))
        episode_grp.create_dataset('rewards', data=np.random.randn(50))
        
        # Create metadata
        f.attrs['num_episodes'] = 1
        f.attrs['max_episode_length'] = 50
    
    return dataset_path


@pytest.fixture
def mock_torch_model():
    """Mock PyTorch model for ACT testing."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        pytest.skip("PyTorch not available for model fixture")
    
    class MockACTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(14, 7)
        
        def forward(self, x):
            return self.linear(x)
    
    model = MockACTModel()
    model.eval()
    return model


@pytest.fixture(autouse=True)
def reset_environment_variables(monkeypatch):
    """Reset environment variables that might affect tests."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MUJOCO_GL", raising=False)


@pytest.fixture
def mock_mujoco_env():
    """Mock MuJoCo environment for simulation tests."""
    mock_env = Mock()
    mock_env.reset.return_value = np.random.randn(14)
    mock_env.step.return_value = (
        np.random.randn(14),  # observation
        0.1,                  # reward
        False,                # done
        {}                    # info
    )
    mock_env.close.return_value = None
    return mock_env


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Directory containing test data files."""
    return Path(__file__).parent / "data"


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture and return log messages."""
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog


# Pytest markers for different test types
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning")
]