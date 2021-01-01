import numpy as np
import matplotlib.pyplot as plt

N = 20  # height of the gridworld ---> number of rows
M = 20  # length of the gridworld ---> number of columns
FONT_SIZE = 18


def homing_nn(n_trials, n_steps, learning_rate, eps, gamma, eps_doff=1, et_lam=0, eight_actions=False,
              multiple_rewards=False):
    """
        The function that used for experimentation in training an agent with Sarsa or (λ) to find reward in the grid.
        :param n_trials: The number of trials for the run
        :param n_steps: The maximum number of steps
        :param learning_rate: The learning rate
        :param eps: The initial epsilon value
        :param gamma: The gamma value
        :param et_lam: The lambda value
        :param eps_dropoff: the dropoff of the epsilon value
        :param eight_actions: A boolean whether it allows eight actions or four
        :param multiple_rewards: A boolean whether there are multiple reward locations or just one
        :return: trial_learning_curve (steps per trial), trial_unique_states (unique states visited per trial)
    """

    # Definition of the environment
    np.set_printoptions(precision=None, suppress=None)
    n_states = N * M  # total number of states
    states_matrix = np.eye(n_states)

    # Initialise actions

    n_actions = 8 if eight_actions else 4  # number of possible actions in each state: 1->n 2->E 3->S 4->W
    action_row_change = np.array([-1, 0, +1, 0, +1, +1, -1, -1])  # number of cell shifted in vertically
    action_col_change = np.array([0, +1, 0, -1, +1, -1, +1, -1])  # number of cell shifted in horizontally

    # Initialise rewards and randomise locations

    n_reward_locations = 5 if multiple_rewards else 1
    current_reward_value = 1
    reward_locations = {}
    for i in range(n_reward_locations):
        current_end = np.random.randint(20, size=(2, 1))  # Terminal state in 2D
        s_end_loc = np.ravel_multi_index(current_end, dims=(N, M), order='F')  # Term state. Single index conversion.
        reward_locations[s_end_loc[0]] = current_reward_value
        current_reward_value += 5

    # Initial variable declarations
    weights = np.random.rand(n_actions, n_states)
    trial_learning_curve = np.zeros((1, n_trials))
    trial_rewards = np.zeros((1, n_trials))
    trial_unique_states = np.zeros((1, n_trials))
    r = 0
    r_old = 0
    total_unique_states = set()

    # Sarsa and Sarsa(λ) implementation
    for trial in range(n_trials):
        # Initial variables setup
        start = np.array([np.random.randint(N), np.random.randint(M)])  # random location as start
        s_start = np.ravel_multi_index(start, dims=(N, M), order='F')  # conversion to single index
        state = start  # set current state to be the starting state
        s_index = s_start  # set current index to be the starting index.
        trace = np.zeros(weights.shape)
        step = 0
        reward_reached = False
        # start steps
        while (not reward_reached) and step <= n_steps:
            step += 1
            total_unique_states.add(s_index)
            input_vector = states_matrix[:, s_index].reshape(n_states, 1)  # convert the state into an input vector
            # compute Qvalues. Q_value=logsig(weights*input). Q_value is 2x1, one value for each output neuron
            Q_value = 1 / (1 + np.exp(- weights.dot(input_vector)))  # Q_value is 2x1 implementation of logsig
            # eps-greedy policy implementation
            greedy = (np.random.rand() > eps)  # 1--->greedy action 0--->non-greedy action
            if greedy:
                action = np.argmax(Q_value)  # pick best action
            else:
                action = np.random.randint(n_actions)  # pick random action

            state_new = np.array([0, 0])
            # move into a new state
            state_new[0] = state[0] + action_row_change[action]
            state_new[1] = state[1] + action_col_change[action]
            # put the robot back in grid if it goes out. Consider also the option to give a negative reward
            if state_new[0] < 0:
                state_new[0] = 0
            if state_new[0] >= N:
                state_new[0] = N - 1
            if state_new[1] < 0:
                state_new[1] = 0
            if state_new[1] >= M:
                state_new[1] = M - 1

            s_index_new = np.ravel_multi_index(state_new, dims=(N, M), order='F')  # conversion in a single index

            # Update Q values if step is not 0
            if step != 1:
                if et_lam == 0:
                    dw = learning_rate * (r_old - Q_value_old + gamma * Q_value[action]) * output_old.dot(input_old.T)
                else:
                    delta = r_old - Q_value_old + gamma * Q_value[action]
                    trace = et_lam * gamma * trace
                    dw = learning_rate * delta * (trace + output_old.dot(input_old.T))
                weights += dw

            # store variables for sarsa computation in the next step
            output = np.zeros((n_actions, 1))
            output[action] = 1

            # update variables
            input_old = input_vector
            output_old = output
            Q_value_old = Q_value[action]
            r_old = r

            state[0] = state_new[0]
            state[1] = state_new[1]
            s_index = s_index_new
            if s_index in reward_locations:  # state is terminal, so set rewards
                r_old = reward_locations[s_index]
                reward_reached = True

        trial_learning_curve[0, trial] = step
        trial_rewards[0, trial] = r_old
        trial_unique_states[0, trial] = len(total_unique_states)
        if et_lam == 0:
            dw = learning_rate * (r_old - Q_value_old) * output_old.dot(input_old.T)
        else:
            delta = r_old - Q_value_old + gamma * Q_value[action]
            dw = learning_rate * delta * (et_lam * gamma * trace + output_old.dot(input_old.T))

        weights += dw
        eps = eps * eps_doff  # Increase exploitation, not exploration

    return trial_learning_curve, trial_unique_states, trial_rewards


def plot_error_graph(data, repetitions, n_trials, n_steps):
    """
            The function that used for experimentation in training an agent with Sarsa or (λ) to find reward in the grid.
            :param data: The data to plot error graph for
            :param repetitions: The number of repetitions the experiment was conducted with.
            :param n_trials: The amount of trials per run
            :param n_steps: Maximum Number of steps. -1 Indicates it is an AVG Reward plot and -2 a Unique States Plot.
    """
    
    if n_steps == -1:  # AVG Reward Plot
        x = 1.1
        y_label = "Average Reward"

    elif n_steps == -2:  # Unique States Discovered Plot
        x = 440
        y_label = "Unique States Discovered (/400)"

    else:
        x = n_steps * 1.1
        y_label = "Average Steps"

    plt.figure(figsize=(8, 6))
    means = np.mean(data, axis=0)
    errors = 2 * np.std(data, axis=0) / np.sqrt(
        repetitions)  # errorbars are equal to twice standard error i.e. std/sqrt(samples)
    plt.errorbar(np.arange(n_trials), means, errors, 0, elinewidth=2, capsize=4, alpha=0.8)
    plt.xlabel("Trials", fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    plt.axis((-(n_trials / 10.0), n_trials, 0.0, x))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()


def main():
    """
        The main function, executes all the experiments here. Alter this function to perform desired experiments.
        Params to be set: n_repetitions, n_trials, n_steps, learning_rate, epsilon, gamma, et_lambda, eps_doff
    """

    print("Setting and Initialising Variables")
    n_repetitions = 20  # number of runs for the algorithm
    n_trials = 200  # should be integer > 0
    n_steps = N * M * 2.5  # maximum number of allowed steps (1000)
    learning_rate = 1  # should be real, Greater than 0
    epsilon = 0.45  # should be real, Greater or Equal to 0; epsilon=0 Greedy, otherwise epsilon-Greedy
    gamma = 0.95  # should be real, positive, smaller than 1
    et_lambda = 0.01  # should be real, positive smaller than 1 (usually really small like 0.01)

    learning_curve = np.zeros((n_repetitions, n_trials))
    trial_rewards = np.zeros((n_repetitions, n_trials))
    total_unique_states = np.zeros((n_repetitions, n_trials))

    print("Conducting multiple runs")
    for i in range(n_repetitions):  # Parameters set to reproduce Question 5 results. Change Accordingly
        learning_curve[i, :], total_unique_states[i, :], trial_rewards[i, :] = homing_nn(n_trials=n_trials,
                                                                                         n_steps=n_steps,
                                                                                         learning_rate=learning_rate,
                                                                                         eps=epsilon, eps_doff=0.925,
                                                                                         gamma=gamma, et_lam=et_lambda,
                                                                                         eight_actions=True)

    print("Plotting the graphs")
    plot_error_graph(learning_curve, n_repetitions, n_trials, n_steps)
    plot_error_graph(total_unique_states, n_repetitions, n_trials, -2)
    plot_error_graph(trial_rewards, n_repetitions, n_trials, -1)


if __name__ == '__main__':
    main()
