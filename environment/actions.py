"""Action definitions and interfaces for Project Caspian.

Actions are represented strictly as neutral numerical enumerated values.
No semantic labels or task-specific designations are embedded in the actions.
"""

from enum import IntEnum
from typing import Tuple, Union


class InvalidActionError(ValueError):
    """Raised when an invalid action is supplied to the environment."""
    pass


class Action(IntEnum):
    """Enumeration of permitted actions in the Caspian environment."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    INTERACT = 4
    NOOP = 5

    @classmethod
    def parse(cls, action: Union["Action", int, str]) -> "Action":
        """Parse and validate an action input.

        Args:
            action: Action instance, integer code, or uppercase string name.

        Returns:
            Action: Validated Action enum member.

        Raises:
            InvalidActionError: If the input cannot be resolved to a valid Action.
        """
        if isinstance(action, cls):
            return action

        if isinstance(action, int) and not isinstance(action, bool):
            try:
                return cls(action)
            except ValueError:
                raise InvalidActionError(
                    f"Action integer {action} is out of valid range [0, {len(cls) - 1}]."
                )

        if isinstance(action, str):
            sanitized = action.strip().upper()
            if sanitized in cls.__members__:
                return cls.__members__[sanitized]
            raise InvalidActionError(
                f"Action string '{action}' is not a recognized action name. "
                f"Permitted: {list(cls.__members__.keys())}."
            )

        raise InvalidActionError(
            f"Invalid action type {type(action).__name__}: expected Action, int, or str."
        )

    def get_delta(self) -> Tuple[int, int]:
        """Compute the coordinate delta (dx, dy) in standard 2D Cartesian grid.

        Returns:
            Tuple[int, int]: (dx, dy) positional offset.
        """
        if self == Action.UP:
            return (0, 1)
        elif self == Action.DOWN:
            return (0, -1)
        elif self == Action.LEFT:
            return (-1, 0)
        elif self == Action.RIGHT:
            return (1, 0)
        elif self == Action.INTERACT or self == Action.NOOP:
            return (0, 0)
        raise InvalidActionError(f"Unhandled action: {self}")
