import autograd.numpy as np
import autograd as ag
from autograd.optimizers import adam

def unit_normal(e):
    """
    Unit normal pdf
    """
    return (1./np.sqrt(2.*np.pi)) * np.exp(-0.5*e**2)

def p(x):
    """
    Unnormalized target distribution, a mixture of 2 Gaussians
    """
    return np.exp(-0.5 * ((x-3.0)**2 + (x+3.0)**2) )

def get_proposal_distribution(x, params):
    """
    MLP with a single hidden layer
    """
    w_h, b_h, w_o, b_o = params
    h = np.maximum(np.dot(x[:, np.newaxis], w_h) + b_h, 0.0)
    o = np.dot(h, w_o) + b_o
    m = o[:, 0]
    # s = np.exp(o[:, 1])
    s = np.ones_like(m)    
    return m, s

def initial_state_distribution():
    return 0.0, 3.0

def reward(es, params):
    """
    Calculate the reward for a given set of samples.
    es are N samples from a T dimensional unit normal used to calculate samples from q.
    """
    px = np.zeros(es.shape[0])
    m_init, s_init = initial_state_distribution()    
    x = m_init + es[:, 0]*s_init
    px = px + p(x)
    for t in range(1, es.shape[1]):
        m, s = get_proposal_distribution(x, params)
        x = m + es[:, t]*s
        px = px + p(x)
    
    qx = np.sum(unit_normal(es), axis=1)
    # importance weights
    wx = px / qx
    # normalize weights
    wx /= np.sum(wx)
    return -np.var(wx)
    
# gradient of the reward function
grad_reward = ag.grad(reward, argnum=1)

def init_weight(*shape):
    return np.random.randn(*shape)*0.1

if __name__ == "__main__":
    hidden_unit_count = 10
    w_h, b_h = init_weight(1, hidden_unit_count), init_weight(hidden_unit_count)
    w_o, b_o = init_weight(hidden_unit_count, 2), init_weight(2)
    init_params = [w_h, b_h, w_o, b_o]
    print "Initial params:", init_params

    chain_length = 2
    samples_per_episode = 20
    episodes_per_epoch = 100
    epoch_count = 10

    learning_rate = 0.001

    def estimate_reward(params, i, reps=20):
        tot_r = 0
        for _ in range(0, reps):
            es = np.random.randn(samples_per_episode, chain_length)
            r = reward(es, params)
            if not np.isnan(r):
                tot_r += r
        return -tot_r / reps

    estimate_grad = ag.grad(estimate_reward, argnum=0)

    def report_progress(params, i, g):
        if (i+1) % episodes_per_epoch == 0:
            print "Iteration", i+1, "reward:", estimate_reward(params, 0, reps=100)

    report_progress(init_params, -1, None)

    new_params = adam(estimate_grad, init_params, callback=report_progress, 
                      num_iters=epoch_count*episodes_per_epoch, step_size=learning_rate)
    print "Learned params:", new_params
