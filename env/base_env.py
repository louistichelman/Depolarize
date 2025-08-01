from abc import ABC, abstractmethod
import copy


class BaseEnv(ABC):
    """
    Base class for all environments used in this project.
    """

    def __init__(self):
        super().__init__()
        self.current_state = None

    @abstractmethod
    def step(self, action, state=None):
        """
        Given either the current_state (if state is None) or a provided state,
        performs a step in the environment with the given action.
        Returns: a tuple (next_state, reward, done).
        """
        raise NotImplementedError("Must be implemented by subclass.")

    @abstractmethod
    def reset(self):
        """
        Resets the current_state to a random starting state.
        Returns: current_state.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def is_terminal(self, state=None):
        """
        Returns True if the current_state (or given state) is terminal, False otherwise.
        """
        return False

    def clone(self):
        """
        Returns a deep copy of the environment.
        This is useful for creating several independent environments for better training.
        """
        return copy.deepcopy(self)
