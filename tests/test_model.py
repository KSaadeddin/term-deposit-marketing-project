import pytest
from src.model import build_pipeline
from src.config import RANDOM_STATE

def test_build_pipeline_structure():
    pipe = build_pipeline(scale_pos_weight=1.0)
    assert 'preprocessor' in pipe.named_steps
    assert 'classifier' in pipe.named_steps
