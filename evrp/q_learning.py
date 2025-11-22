


def get_best_action(Q, state, actions):
    if state not in Q or not Q[state]:
        return actions[0]
    return max(Q[state].items(), key=lambda kv: kv[1])[0]

def update(Q, state, action, cost, next_state, alpha, gamma, actions):
    """
    Q-learning update for cost minimization (inf/nan safe).
    Lower cost = better.
    """
    import math
    if not math.isfinite(cost):
        return  # skip invalid updates

    old_value = Q.get((state, action), 0.0)
    next_min = min(Q.get((next_state, a), float('inf')) for a in actions)
    if not math.isfinite(next_min):
        next_min = 0.0

    target = cost + gamma * next_min
    if not math.isfinite(target):
        return

    Q[(state, action)] = old_value + alpha * (target - old_value)

def decay_epsilon(eps, eps_min, decay):
    new_eps = eps * decay
    return eps_min if new_eps < eps_min else new_eps
