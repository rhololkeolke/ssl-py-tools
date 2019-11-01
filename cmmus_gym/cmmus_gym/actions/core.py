"""Common classes used by different ActionClients."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ActionClientStats:
    num_sent: int = 0
    last_set_action_time: Optional[float] = None
    last_action_sent_time: Optional[float] = None
