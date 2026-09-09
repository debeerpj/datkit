"""Shared pytest fixtures.

pytest discovers this file automatically — nothing imports it.
"""

import pandas as pd
import pytest

# The three string representations the toolkit may meet. The target environment
# decides which one it gets, so the cleaning functions must behave identically
# on all three.
_TEXT_DTYPES = ["object", "string"]

try:
    import pyarrow as pa

    _TEXT_DTYPES.append(pd.ArrowDtype(pa.string()))
except ImportError:  # pragma: no cover - only when pyarrow is absent
    pass


@pytest.fixture(params=_TEXT_DTYPES, ids=lambda d: str(d))
def text_dtype(request):
    """Run a test once per available string backend."""
    return request.param
