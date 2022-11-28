from src.agents.random_agent import RandomAgent
from src.agents.dueling_dqn import DuelingDQN
from src.agents.ddqn import DDQN
from src.agents.dqn import DQN

from src.estimators import NNEstimator, DuelingNNEstimator
from src.replay_buffer import ReplayBuffer

from typing import Optional, Callable

import itertools
import torch



def get_agent(
    env,
    opt,
    agent: str = 'dqn',
    replay_buffer_size: int = 10_000,
    batch_size: int = 32,
    lr: float = 6.25e-4,
    eps: float = 1e-5,
    start: float = 1.0,
    end: float = 0.1,
    warmup_steps_percent: float = 0.1,
    update_steps: int = 1,
    update_target_steps: int = 4,
    estimator: Optional[Callable] = None,
    optimizer: Optional[Callable] = None,
    epsilon_schedule: Optional[Callable] = None
):

    device = opt.device
    action_num = env.action_space.n
    warmup_steps = opt.steps * warmup_steps_percent

    if estimator is None:
        if agent == 'dueling':
            estimator = NNEstimator(action_num).to(device)
        else:
            estimator = DuelingNNEstimator(action_num).to(device)


    if optimizer is None:
        optimizer = torch.optim.Adam(estimator.parameters(), lr=lr, eps=eps)

    if epsilon_schedule is None:
         epsilon_schedule = get_epsilon_schedule(start=start, end=end, steps=opt.steps)

    buffer = ReplayBuffer(device, size=replay_buffer_size, batch_size=batch_size)


    if agent == 'dqn':
        agent = DQN(estimator, buffer, optimizer, epsilon_schedule, action_num,
                    warmup_steps=warmup_steps, update_steps=update_steps)

    elif agent == 'ddqn':
        agent =  DDQN(estimator, buffer, optimizer, epsilon_schedule, action_num,
                      warmup_steps=warmup_steps, update_steps=update_steps, update_target_steps=update_target_steps)


    elif agent == 'dueling':
        agent = DuelingDQN(estimator, buffer, optimizer, epsilon_schedule, action_num,
                          warmup_steps=warmup_steps, update_steps=update_steps,
                          update_target_steps=update_target_steps)

    elif agent == 'random':
        agent = RandomAgent(action_num)

    else:
        # if no valid agent type was chosen then return random agent
        agent = RandomAgent(action_num)

    print(f"Agent: {agent}")

    return agent

def get_epsilon_schedule(start: float = 1.0, end: float = 0.1, steps: int = 500):

    eps_step = (start - end) / steps
    def frange(start, end, step):
        x = start
        while x > end:
            yield x
            x -= step
    return itertools.chain(frange(start, end, eps_step), itertools.repeat(end))
