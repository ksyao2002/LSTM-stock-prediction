import numpy as np
import random
from enum import IntEnum

class Action(IntEnum):
    BUY = 0
    SELL = 1
    HOLD = 2

class Agent:
    def __init__(self):
        """
        Write your custom initialization sequence here.
        This can include loading models from file.
        """
        self.useless_var = 0

    def step(self, row):
        """
        Make a decision to be executed @ the open of the next timestep. 

        row is a numpy array with the same format as the training data

        Return a tuple (Action, fraction). Fraction means different 
        things for different actions...
        
        Action.BUY:  represents fraction of cash to spend on purchase 
        Action.SELL: represents fraction of owned shares to sell 
        Action.HOLD: value ignored.

        See the code below on how to return
        """

        choice = random.randint(0, 2)

        # The plan was to never have to use constants...
        # Yeah, we're assuming consistency in buy=0, sell=1, and hold=2
        if choice == 0:
            return (Action.BUY, 1)
        elif choice == 1:
            return (Action.SELL, 1)
        elif choice == 2:
            return (Action.HOLD, 1)
        else:
            raise ValueError(f"Choice was odd value: choice={choice}")