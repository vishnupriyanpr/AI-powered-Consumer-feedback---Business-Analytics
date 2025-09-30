"""Simple configuration for AMIL Project"""

import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent

# GPU Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 if DEVICE == "cuda" else 8

# API settings
API_HOST = "localhost"
API_PORT = 5000
DEBUG_MODE = True

print(f"🔧 Config loaded - Device: {DEVICE}")
