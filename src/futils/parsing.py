"""Parse the filename of the output of a measurement simulated in rust.
Rust saves the output of the simulations by encoding the parameters used to
simulate into the filename.
Here we want to extract the parameters from the filenames.
Assume the file `myfile.myext` is saved as:
    `/path/to/dir/{number}cells/{measurement}/myfile.myext`
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, NewType, Union

Parameters = NewType("Parameters", Dict[str, Union[int, float]])


class SampleSizeIsZero(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code


class SampleSizeNotParsed(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code


def find_sample_size(path: Path) -> int:
    match_sample = re.compile(r"^(\d+)(cells)$", re.IGNORECASE)
    parts = path.parts
    for part in parts:
        matched = match_sample.search(part)
        if matched:
            # assume the first (\d+)(cells) is the sample size
            sample_size = int(matched.group(1))
            # neg sample size wont match
            if sample_size == 0:
                raise SampleSizeIsZero(
                    f"Found sample size of 0 from file {path}", error_code=1
                )
            return sample_size
    raise SampleSizeNotParsed(
        f"Cannot match regex to parse the sample size from file {path}",
        error_code=1,
    )


def parameters_from_path(path: Path) -> Parameters:
    """The main method to use: take a path as input and returns a dict.
    The path must follow the convention:
    `/path/to/dir/{number}cells/{measurement}/myfile.myext`
    """
    params_file = parameters_from_filename(path)
    params_file["cells"] = find_sample_size(path)
    return Parameters(params_file)


def parameters_from_filename(filename: Path) -> Parameters:
    match_nb = re.compile(r"(\d+\.?\d*)([a-z]+\d*)", re.IGNORECASE)
    filename_str = filename.stem
    filename_str = filename_str.replace("dot", ".").split("_")
    my_dict = dict()
    for ele in filename_str:
        matched = match_nb.search(ele)
        if matched:
            if matched.group(2) == "idx":
                my_dict[matched.group(2)] = int(matched.group(1))
            else:
                my_dict[matched.group(2)] = float(matched.group(1))
        else:
            raise ValueError(
                f"Syntax <NUMBER><VALUE>_ not respected in filename {filename}"
            )
    return Parameters(my_dict)


def params_into_dataframe(params: List[Parameters]) -> pd.DataFrame:
    df = pd.DataFrame.from_records([param for param in params])
    df.idx = df.idx.astype(int)
    df.cells = df.cells.astype(int)
    df["samples"] = df["samples"].astype(int)
    return df
