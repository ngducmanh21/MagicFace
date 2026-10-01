"""AU ordering and validation shared by inference and verification."""

import math


AU_NAMES = tuple(f"AU{i}" for i in (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26))


def edit_metadata(values):
    """Describe an AU request for stable filenames, tables and fair grouping."""
    active = [name for name, value in values.items() if float(value) != 0]
    if not active:
        return {'edit_type': 'zero_baseline', 'active_aus': [], 'edit_au': None, 'edit_level': 0.0}
    if len(active) == 1:
        name = active[0]
        return {'edit_type': 'single_au', 'active_aus': active,
                'edit_au': name, 'edit_level': float(values[name])}
    return {'edit_type': 'combination', 'active_aus': active, 'edit_au': None, 'edit_level': None}


def validate_request(values):
    if not isinstance(values, dict) or not values:
        raise ValueError("requested_aus must be a non-empty mapping, e.g. {'AU4': 2}.")
    unknown = set(values) - set(AU_NAMES)
    if unknown:
        raise ValueError(f"Unsupported AUs: {sorted(unknown)}. Supported: {', '.join(AU_NAMES)}")
    result = {}
    for name, value in values.items():
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number.")
        result[name] = number
    return result


def parse_au_request(names, variations):
    keys = [name.strip().upper() for name in names.split('+')]
    values = variations.split('+')
    if len(keys) != len(values):
        raise ValueError("--au_test and --AU_variation must have the same number of entries.")
    if len(set(keys)) != len(keys):
        raise ValueError("Each AU may appear only once in --au_test.")
    return validate_request(dict(zip(keys, values)))


def normalize_intensities(values):
    """Accept only the 12 intensity outputs, never binary AU detections."""
    result = {}
    for name in AU_NAMES:
        key = f"au_{name[2:]}_intensity"
        value = float(values[key])
        if not math.isfinite(value) or not 0 <= value <= 5:
            raise ValueError(f"Invalid LibreFace intensity {key}: {value}; expected 0..5.")
        result[name] = value
    return result
