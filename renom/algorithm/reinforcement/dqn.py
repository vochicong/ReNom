
from __future__ import division
from time import sleep
import copy
from tqdm import tqdm
import numpy as np
import renom as rm
from renom.utility.reinforcement.replaybuffer import ReplayBuffer


class DQN(object):

    def __init__(self, q_network, state_size=10, action_pattern=10, ganma=0.99, buffer_size=1e5):
        self._network = q_network
        self._target_network = copy.deepcopy(q_network)
        self._action_size = action_pattern
        self._state_size = state_size if hasattr(state_size, "__getitem__") else [state_size, ]
        self._buffer_size = buffer_size
        self._ganma = ganma
        self._buffer = ReplayBuffer([1, ], self._state_size, buffer_size)

    def action(self, state):
        self._network.set_models(inference=True)
        shape = [-1, ] + list(self._state_size)
        s = state.reshape(shape)
        return np.argmax(self._network(s).as_ndarray(), axis=1)

    def update(self):
        # Check GPU data
        self._target_network = copy.deepcopy(self._network)
        for n, target_n in zip(self._network.iter_models(), self._target_network.iter_models()):
            if hasattr(n, "params") and hasattr(target_n, "params"):
                for k in n.params.keys():
                    target_n.params[k] = rm.Variable(n.params[k])

    def train(self, env, loss_func=rm.ClippedMeanSquaredError(), optimizer=rm.Rmsprop(lr=0.00025, g=0.95),
              episode=100, batch_size=32, random_step=1000, one_episode_step=20000, test_step=1000,
              test_env=None, update_period=10000, greedy_step=1000000, min_greedy=0.0, train_frequency=4,
              max_greedy=0.9, test_greedy=0.95, callbacks=None):

        greedy = min_greedy
        g_step = (max_greedy - min_greedy) / greedy_step

        if test_env is None:
            test_env = env

        print("Execute random action for %d step..." % random_step)
        for r in range(random_step):
            action = int(np.random.rand() * self._action_size)
            prestate, action, reward, state, terminal = env(action)
            if prestate is not None:
                self._buffer.store(prestate, np.array(action),
                                   np.array(reward), state, np.array(terminal))

        state = None
        prestate = None
        count = 0
        for e in range(episode):
            loss = 0
            sum_reward = 0
            tq = tqdm(range(one_episode_step))
            for j in range(one_episode_step):
                if greedy > np.random.rand() and state is not None:
                    action = np.argmax(np.atleast_2d(self._network(state[None, ...])), axis=1)
                else:
                    action = int(np.random.rand() * self._action_size)
                prestate, action, reward, state, terminal = env(action)
                greedy += g_step
                greedy = np.clip(greedy, min_greedy, max_greedy)
                sum_reward += reward
                if prestate is not None:
                    self._buffer.store(prestate, np.array(action),
                                       np.array(reward), state, np.array(terminal))

                if j % train_frequency == 0:
                    # Training
                    train_prestate, train_action, train_reward, train_state, train_terminal = \
                        self._buffer.get_minibatch(batch_size)

                    target = np.zeros((batch_size, self._action_size), dtype=state.dtype)
                    for i in range(batch_size):
                        target[i, train_action[i, 0].astype(np.integer)] = train_reward[i]
                    try:
                        target += (self._target_network(train_state).as_ndarray() *
                                   self._ganma * (~train_terminal[:, None]))
                    except Exception as e:
                        raise e

                    self._network.set_models(inference=False)
                    with self._network.train():
                        z = self._network(train_prestate)
                        l = loss_func(z, target)
                    l.grad().update(optimizer)
                    loss += l.as_ndarray()

                    if count % update_period == 0:
                        self.update()
                        count = 0
                    count += 1
                msg = "episode {:03d} loss:{:6.4f} sum reward:{:5.3f}".format(
                    e, float(l.as_ndarray()), sum_reward)
                tq.set_description(msg)
                tq.update(1)
            msg = ("episode {:03d} avg loss:{:6.4f} avg reward:{:5.3f}".format(
                e, float(loss) / (j + 1), sum_reward / one_episode_step))
            tq.set_description(msg)
            tq.update(0)
            tq.refresh()
            tq.close()

            # Test
            state = None
            sum_reward = 0
            for j in range(test_step):
                if test_greedy > np.random.rand() and state is not None:
                    action = self.action(state)
                else:
                    action = int(np.random.rand() * self._action_size)
                prestate, action, reward, state, terminal = test_env(action)
                sum_reward += float(reward)

            tq.write("    /// Result")
            tq.write("    Average train error:{}".format(float(loss) / one_episode_step))
            tq.write("    Test reward:{}".format(sum_reward))
            tq.write("    Greedy:{:1.4f}".format(greedy))
            tq.write("    Buffer:{}".format(len(self._buffer)))

            if isinstance(callbacks, dict):
                func = callbacks.get("end_episode", False)
                if func:
                    func()

            sleep(0.25)  # This is for jupyter notebook representation.
