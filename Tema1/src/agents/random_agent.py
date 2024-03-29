import torch


class RandomAgent:
    """An example Random Agent"""

    def __init__(self, action_num) -> None:
        self.action_num = action_num
        # a uniformly random policy
        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )

    def act(self, observation):
        """ Since this is a random agent the observation is not used."""
        return self.policy.sample().item()

    def step(self, state):
        return self.policy.sample().item()

    def learn(self, state, action, reward, state_, done):
        pass

    def __str__(self):
        return 'Random Agent'