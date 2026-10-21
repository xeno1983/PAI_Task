"""Go2 + Franka pick-and-carry environment for the Genesis simulator."""

from pick_and_carry.config import NUM_ACTIONS, NUM_OBS

__all__ = ["NUM_ACTIONS", "NUM_OBS", "PickAndCarryEnv"]


def __getattr__(name: str):
    if name == "PickAndCarryEnv":
        from pick_and_carry.env import PickAndCarryEnv

        return PickAndCarryEnv
    raise AttributeError(name)
