"""Common classes/functions between different observation types."""


from dataclasses import dataclass
from typing import Optional


@dataclass
class ObservationStats:
    num_updates: int = 0
    last_update_time: Optional[float] = None
