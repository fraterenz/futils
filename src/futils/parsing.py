"""Parse the filename of the output of a measurement simulated in rust.
Rust saves the output of the simulations by encoding the parameters used to
simulate into the filename.
Here we want to extract the parameters from the filenames.
Assume the file `myfile.myext` is saved as:
    `/path/to/dir/{number}cells/{measurement}/myfile.myext`
"""

import re
from pathlib import Path
from typing import Dict, Union


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
        print(part)
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


def parameters_from_path(path: Path) -> Dict[str, Union[int, float]]:
    """The main method to use: take a path as input and returns a dict.
    The path must follow the convention:
    `/path/to/dir/{number}cells/{measurement}/myfile.myext`
    """
    params_file = parse_filename_into_dict(path)
    params_file["cells"] = find_sample_size(path)
    return params_file


def parse_filename_into_dict(filename: Path) -> Dict[str, Union[int, float]]:
    match_nb = re.compile(r"(\d+\.?\d*)([a-z]+\d*)", re.IGNORECASE)
    filename_str = filename.stem
    filename_str = filename_str.replace("dot", ".").split("_")
    my_dict = dict()
    for ele in filename_str:
        matched = match_nb.search(ele)
        if matched:
            my_dict[matched.group(2)] = float(matched.group(1))
        else:
            raise ValueError(
                f"Syntax <NUMBER><VALUE>_ not respected in filename {filename}"
            )
    return my_dict
