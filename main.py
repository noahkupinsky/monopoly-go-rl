from functools import reduce

# states are a pair: board square, dice rolls left
board = [1, 1, 5, 1, 1, 10, 1, 1, 5, 1]  # railroads 10, everything else 1
size = len(board)
board_max = max(board)
roll_probabilities = list(map((lambda x: x / 36.0), [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]))
roll_max = 1000
state_max = size * roll_max
multipliers = [1, 2, 3, 5, 10, 20, 50, 100]


def square(i):
    return board[i % size]


def rolls(s):
    return s % roll_max + 1


def position(s):
    return s // roll_max


def state(state_position, state_rolls):
    return (state_position % size) * roll_max + (state_rolls - 1)


def num_multipliers(x):
    if x <= 1:
        return 1
    elif x <= 2:
        return 2
    elif x < 50:
        return 3
    elif x < 100:
        return 4
    elif x < 300:
        return 5
    elif x < 1000:
        return 6
    elif x < 2000:
        return 7
    else:
        return 8


def init_policy():
    policy = []
    for s in range(state_max):
        m = num_multipliers(rolls(s))
        action_ps = [1.0 / m] * m  # make all allowed multipliers equally likely
        policy.append(action_ps)
    return policy


def reward(s, a):
    return multipliers[a] * reduce(lambda t, d: t + roll_probabilities[d] * square(position(s) + d + 2), range(11), 0)


def transition_states(s, a):
    rol = rolls(s)
    if rol - multipliers[a] <= 0:
        return []
    pos = position(s)
    return list(map((lambda d: state(pos + 2 + d, rol - multipliers[a])), range(11)))


def reward_and_sum(v, s, a, verbose=False):
    t = reward(s, a)
    ts = transition_states(s, a)
    if verbose:
        for z in ts:
            print("position: " + str(position(z)) + ", rolls: " + str(rolls(z)))
    for i in range(len(ts)):
        t += roll_probabilities[i] * v[ts[i]]
    return t


def policy_eval(pi, v):
    return list(map((lambda s: reduce((lambda t, a: t + pi[s][a] * reward_and_sum(v, s, a)), range(len(pi[s])), 0)), range(state_max)))


def policy_update(v):
    pi = []
    for s in range(state_max):
        m = num_multipliers(rolls(s))
        r_max = -1
        a_max = -1
        for a in range(m):
            r = reward_and_sum(v, s, a)
            if r > r_max:
                r_max = r
                a_max = a
        actions = [0] * m
        actions[a_max] = 1
        pi.append(actions)
    return pi


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    p = init_policy()
    v = policy_eval(p, [0] * state_max)
    while (pp := policy_update(v)) != p:
        v = policy_eval(p, v)
        p = pp

    def mm(i, n):
        k = 0
        while p[state(i, n)][k] == 0:
            k += 1
        return multipliers[k]
    l = []
    for y in range(roll_max):
        ll = list(map(lambda x: mm(x, y+1), range(10)))
        if ll != l:
            print(str(y + 1) + ": " + str(ll))
        l = ll


