from utils import *


class Worker:
    def __init__(self, id, state_shape, env_name, max_episode_steps):
        self.id = id
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.state_shape = state_shape
        self.env = make_atari(self.env_name, self.max_episode_steps)
        self.lives = self.env.ale.lives()
        self._stacked_states = np.zeros(self.state_shape, dtype=np.uint8)
        self.reset()

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        self._stacked_states = stack_states(self._stacked_states, state, True)
        self.lives = self.env.ale.lives()

    def step(self, conn):
        t = 1
        while True:
            conn.send(self._stacked_states)
            action = conn.recv()
            next_state, r, d, info = self.env.step(action)
            t += 1
            if t % self.max_episode_steps == 0:
                d = True
            # self.render()
            self._stacked_states = stack_states(self._stacked_states, next_state, False)
            conn.send((self._stacked_states, np.sign(r), d, info))
            if d:
                self.reset()
                t = 1
