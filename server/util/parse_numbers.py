#
# parse_numbers.py
# Bart Trzynadlowski
#
# Helpers for parsing numeric values from strings.
#

def try_parse_int(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        pass
    return None

