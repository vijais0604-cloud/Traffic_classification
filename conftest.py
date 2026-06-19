import os
import sys

# Ensure repository root is on sys.path so local packages like deployment are importable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
