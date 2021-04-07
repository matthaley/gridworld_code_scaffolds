# Implements the Monte Carlo algorithm (without exploring starts), as described in Sutton & Barto 2nd Ed, p.101.
import operator

import numpy as np

import gridworld_gym_env
from gridworld_gym_env import standard_grid
from gridworld_gym_env import print_values, print_policy

GAMMA = 0.9


def get_epsilon_greedy_action(a, epsilon=0.1):
    """Implement epsilon greedy decision
    ENTER CODE HERE
    """


def play_episode(env, policy):
    # returns a list of states, actions, and returns corresponding to the game played
    s = env.reset()
    a = get_epsilon_greedy_action(policy[s])

    # be aware of the timing: each triple is s(t), a(t), r(t)
    # but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t).
    # the first state has no reward since we didn't arrive there by a previous action
    r = 0
    states_actions_rewards = [(s, a, r)]
    done = False
    while not done:
        s, r, done, _ = env.step(a)
        if env.game_over():
            # no further actions are taken from the terminal state
            a = None
        else:
            a = get_epsilon_greedy_action(policy[s])  # the next state is stochastic
        states_actions_rewards.append((s, a, r))

    # calculate the returns by working backwards from the terminal state
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless since it doesn't correspond to any move
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA * G
    states_actions_returns.reverse()  # we want it to be in order of state visited
    return states_actions_returns


if __name__ == '__main__':
    env = standard_grid()

    # state -> action
    # initialize a random policy
    policy = {}
    for s in env.actions.keys():
        policy[s] = np.random.choice(gridworld_gym_env.ACTION_SPACE)

    # initialize Q(s,a) and returns
    Q = {}
    returns = {}  # dictionary of state -> list of returns we've received
    states = env.all_states()
    for s in states:
        if s in env.actions:  # not a terminal state
            Q[s] = {}
            for a in gridworld_gym_env.ACTION_SPACE:
                Q[s][a] = 0
                returns[(s, a)] = []
        else:
            # terminal state or state we can't otherwise get to
            pass

    # repeat a large number of times until convergence is reached
    deltas = []
    for t in range(5000):
        if t % 1000 == 0:
            print('Iteration %d' % t)

        """generate an episode using the current policy, 
        update the returns for every state-action pair,
        calculate Q for each state-action pair,
        update the policy at every state using pi(s) = argmax[a]{ Q(s,a) }
        ENTER CODE HERE
        """

    # find the optimal state-value function
    # V(s) = max[a]{ Q(s,a) }
    V = {s: Q[s][policy[s]] for s in Q.keys() if s in env.actions.keys()}

    print("final values:")
    print_values(V, env)
    print("final policy:")
    print_policy(policy, env)
