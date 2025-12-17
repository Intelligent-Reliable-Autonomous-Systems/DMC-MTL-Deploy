"""
util.py

Utility Files for training algorithms and data processing

Written by Will Solow, 2025
"""

import torch
import torch.nn as nn
import numpy as np
import re


def assert_yyyy_mm_dd(dates:np.ndarray) -> np.ndarray:
    """
    Assert that dates follow yyyy-mm-dd format 
    
    :param dates: Description
    """
    _iso_date = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    for d in dates.flat:
        s = str(d)
        if not _iso_date.match(s):
            raise ValueError(f"Invalid date format (expected YYYY-MM-DD): {s}")

    # Optional: ensure they are *valid* calendar dates
    try:
        return dates.astype("datetime64[D]")
    except Exception:
        raise ValueError(f"Invalid calendar date in {dates}")
    
def set_embedding_op(model: nn.Module) -> int:
    """
    Set the embedding operation to be used
    in MultiTask Embedding Models
    """

    def concat(embed, input):
        return torch.concatenate((embed, input), dim=-1)

    model.embed_op = concat
    return 2 * model.input_dim
