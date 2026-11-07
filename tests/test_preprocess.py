import pytest
import pandas as pd
from unittest.mock import mock_open, patch
from src.preprocess import repair_glued_csv_text

def test_repair_glued_csv():
    bad_text = "campaign,y58,management\n"
    good_text = repair_glued_csv_text(bad_text)
    assert "campaign,y\n58" in good_text
