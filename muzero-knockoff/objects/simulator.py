import time 
import numpy as np
import random as rand
import threading
import os

class SnakeGame1D:
    def __init__(self, xlen = 10):
        self.state = np.zeros(xlen)
        self.interval = 0.5
        self.score = 0
        self.direction = rand.choice([-1,1])
        self.actions = ["L", "R", "NOP"]
        self.pos = xlen // 2
        self.fruits = [rand.randint(0, xlen - 1)]
        self.running = True

        for fruit in self.fruits:
            self.state[fruit] = 2
        self.state[self.pos] = 1

    def step(self, action) -> int:
        reward = 0
        if action == self.actions[0]:
            self.direction = -1
        if action == self.actions[1]:
            self.direction = 1
        
        self.state[self.pos] = 0
        self.pos += self.direction

        if self.pos < 0 or self.pos >= len(self.state):
            os.system("clear")
            print(f"\nGame over - Score: {self.score}\n")
            return self.state, -100, True
        
        if self.pos in self.fruits:
            self.score += 1
            reward = 1
            self.fruits.remove(self.pos)
            self.fruits.append(rand.randint(0, len(self.state) - 1))
        for fruit in self.fruits:
            self.state[fruit] = 2
        self.state[self.pos] = 1

        return self.state, reward, False

    def get_state(self):
        return self.state

    def action_space(self):
        return self.actions

    def render(self):
        os.system("clear")
        output = f"\nScore: {self.score} \n\n|"
        for i in range(len(self.state)):
            if i == self.pos:
                output += "O|"
                continue
            if i in self.fruits:
                output += "|"
                continue
            output += " |"
        print(output)


game = SnakeGame1D()
actons = game.action_space()
total_reward = 0
while True:
    state, reward, term = game.step(rand.choice(actons))
    total_reward += reward
    if term:
        break
    game.render()
    time.sleep(0.3)
print(f"Total reward: {total_reward}")
# input_thread = threading.Thread(target=game.capture_keyboard_input, daemon=True)  # Run input in background
# input_thread.start()


