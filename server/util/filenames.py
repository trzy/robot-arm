#
# filenames.py
# Bart Trzynadlowski
#
# Functions for automatically generating filenames.
#

import os

from .parse_numbers import try_parse_int


def get_next_numbered_dirname(prefix: str, root_dir: str) -> str:
    """
    Generates a directory name with the format: root_dir/prefix-N, where N is one higher than the
    currently highest numbered directory.

    Parameters
    ----------
    prefix : str
        Directory name prefix. E.g., "foo" will produce "foo-0", "foo-1", etc.
    root_dir : str
        Base directory within which this directory will exist. E.g., "x/y/z" -> "x/y/z/foo-0",
        "x/y/z/foo-1", etc.
    
    Returns
    -------
    str
        The next directory name, with root_dir included (e.g., "x/y/foo-2").
    """
    next_number = 0
    if os.path.exists(root_dir):
        existing_dirs = [ dir for dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, dir)) ]
        existing_dirs = [ dir for dir in existing_dirs if dir.startswith(prefix) ]
        dirs_numbers = [ dir.split("-") for dir in existing_dirs ]
        numbers = [ try_parse_int(pair[1]) for pair in dirs_numbers if len(pair) == 2 ]
        numbers = sorted([ number for number in numbers if number is not None ])
        next_number = (numbers[-1] + 1) if len(numbers) >= 1 else 0
    return os.path.join(root_dir, f"{prefix}-{next_number}")


    