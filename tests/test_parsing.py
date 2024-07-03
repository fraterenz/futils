import pytest
from pathlib import Path
from src.futils import parsing


def test_neg_sample_size():
    path = Path("/path/to/-1cells")
    with pytest.raises(
        parsing.SampleSizeNotParsed,
        match=f"Cannot match regex to parse the sample size from file {path}",
    ):
        parsing.find_sample_size(path)


def test_zero_sample_size():
    path = Path("/path/to/0cells")
    with pytest.raises(
        parsing.SampleSizeIsZero, match=f"Found sample size of 0 from file {path}"
    ):
        parsing.find_sample_size(path)


def test_sample_size():
    path = Path("/path/to/10cells/yo")
    assert parsing.find_sample_size(path) == 10


def test_big_sample_size():
    path = Path("/path/to/10000cells")
    assert parsing.find_sample_size(path) == 10**4


def test_parse_filename_into_dict_only_dir():
    path = Path("/path/to/10000cells/sfs/")
    with pytest.raises(ValueError):
        parsing.parse_filename_into_dict(path)


def test_parse_filename_into_dict_missing_value():
    path = Path("/path/to/10000cells/sfs/300")
    with pytest.raises(ValueError):
        parsing.parse_filename_into_dict(path)


def test_parse_filename_into_dict_missing_extension():
    path = Path("/path/to/10000cells/sfs/300rate_")
    with pytest.raises(ValueError):
        parsing.parse_filename_into_dict(path)


def test_parse_filename_into_dict_trailing_sep():
    path = Path("/path/to/10000cells/sfs/300rate_.json")
    with pytest.raises(ValueError):
        parsing.parse_filename_into_dict(path)


def test_parse_filename_into_dict_int():
    path = Path("/path/to/10000cells/sfs/300rate.json")
    assert parsing.parse_filename_into_dict(path) == {"rate": 300}


def test_parse_filename_into_dict_float_int():
    path = Path("/path/to/10000cells/sfs/300dot1rate_20mu.json")
    assert parsing.parse_filename_into_dict(path) == {"rate": 300.1, "mu": 20}


def test_parse_filename_into_dict_float():
    path = Path("/path/to/10000cells/sfs/300dot1rate.json")
    assert parsing.parse_filename_into_dict(path) == {"rate": 300.1}


def test_parse_filename_into_dict_borderline():
    path = Path("/path/to/10000cells/entropy/300dot32re1_20nu.json")
    assert parsing.parse_filename_into_dict(path) == {"re1": 300.32, "nu": 20}
