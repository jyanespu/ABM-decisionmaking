import mesa
import numpy as np
import random


class AgentCoor(mesa.Agent):
    def __init__(self, unique_id, model, pnb, B, C, update_amount=0.01):
        """
        Initializes the agent for the Coordination game

        :param unique_id: A unique ID for the agent
        :param model: The current team being runned
        :param pnb: The personal normative belief of the agent ( normal uniform etc. )
        :param B: The initializations of the coefficients of the utility function ( normal constant etc. )
        :param C: The game-specific parameter
        :param update_amount: The learning update amount for RL
        """
        super().__init__(unique_id, model)
        self.action = np.random.normal(loc=0.5)

        if pnb == "normal":
            self.pnb = np.random.normal(loc=0.5)
        elif pnb == "uniform":
            self.pnb = np.random.uniform()

        self.ee = self.pnb
        self.ne = self.pnb

        if B == "normal":
            self.B = np.array(
                [
                    np.random.normal(loc=0.5),
                    np.random.normal(loc=0.5),
                    np.random.normal(loc=0.5),
                    np.random.normal(loc=0.5),
                ]
            )
        elif B == "constant":
            self.B = np.array([0.2, 0.2, 0.2, 0.2])

        self.weight = np.zeros_like(self.B)
        self.C = C
        self.update_amount = update_amount
        self.update_utility()

    def utility_function(self):
        return (
            self.B[0] * self.weight[0]
            + self.B[1] * self.weight[1]
            + self.B[2] * self.weight[2]
            + self.B[3] * self.weight[3]
        )

    def update_utility(self):
        self.utility = self.utility_function()

    def update_coeffs(self):
        # Compute the weight of the utility function with respect to each coefficient
        self.weight[0] = 0.1 - self.C * (self.action - self.ee) ** 2
        self.weight[1] = -0.5 * (self.action - self.pnb) ** 2
        self.weight[2] = -0.5 * (self.action - self.ne) ** 2
        self.weight[3] = -0.5 * (self.action - self.ee) ** 2

        # Find the index of the coefficient with the largest absolute weight magnitude
        max_index = np.argmax(np.abs(self.weight))

        # Update the selected coefficient slightly
        self.B[max_index] += self.update_amount

        # Ensure that all coefficients sum up to 1
        self.B /= np.sum(self.B)

        # Update the utility function
        self.update_utility()

    def update_action(self):
        # Update self.action with the action that maximizes the utility function
        self.action = (
            2 * self.C * self.B[0] * self.ee
            + self.B[1] * self.pnb
            + self.B[2] * self.ne
            + self.B[3] * self.ee
        ) / (2 * self.C * self.B[0] + self.B[1] + self.B[2] + self.B[3])

        if self.action > 1:
            self.action = 1
        elif self.action < 0:
            self.action = 0

    def update_empirical(self):
        sum_action_values = 0
        sum_pnb_values = 0
        for agent in self.model.schedule.agents:
            # Exclude the i-th agent
            if agent != self:
                sum_action_values += agent.action
                sum_pnb_values += agent.pnb
        self.ee = sum_action_values / (len(self.model.schedule.agents) - 1)
        self.ne = sum_pnb_values / (len(self.model.schedule.agents) - 1)

    def step(self):
        self.update_empirical()
        self.update_coeffs()
        self.update_action()


class AgentCPD(mesa.Agent):
    def __init__(self, unique_id, model, pnb, B, C, update_amount=0.01):
        """
        Initializes the agent for the Continuous Prisoner's Dilemma game

        :param unique_id: A unique ID for the agent
        :param model: The current team being runned
        :param pnb: The personal normative belief of the agent ( normal uniform etc. )
        :param B: The initializations of the coefficients of the utility function ( normal constant etc. )
        :param C: The game-specific parameter
        :param update_amount: The learning update amount for RL
        """
        super().__init__(unique_id, model)
        self.action = np.random.normal(loc=0.5)

        if pnb == "normal":
            self.pnb = np.random.normal(loc=0.5)
        elif pnb == "uniform":
            self.pnb = np.random.uniform()

        self.ee = self.pnb
        self.ne = self.pnb

        if B == "normal":
            self.B = np.array(
                [
                    np.random.normal(loc=0.5),
                    np.random.normal(loc=0.5),
                    np.random.normal(loc=0.5),
                    np.random.normal(loc=0.5),
                ]
            )
        elif B == "constant":
            self.B = np.array([0.2, 0.2, 0.2, 0.2])

        self.weight = np.zeros_like(self.B)
        self.C = C
        self.update_amount = update_amount
        self.pair_action = np.random.uniform()
        self.teammates = []
        self.update_utility()

    def utility_function(self):
        return (
            self.B[0] * self.weight[0]
            + self.B[1] * self.weight[1]
            + self.B[2] * self.weight[2]
            + self.B[3] * self.weight[3]
        )

    def update_utility(self):
        self.utility = self.utility_function()

    def update_coeffs(self):
        # Compute the weight of the utility function with respect to each coefficient
        self.weight[0] = self.pair_action - self.C * self.action
        self.weight[1] = -0.5 * (self.action - self.pnb) ** 2
        self.weight[2] = -0.5 * (self.action - self.ne) ** 2
        self.weight[3] = -0.5 * (self.action - self.ee) ** 2

        # Find the index of the coefficient with the largest absolute weight magnitude
        max_index = np.argmax(np.abs(self.weight))

        # Update the selected coefficient slightly
        self.B[max_index] += self.update_amount

        # Ensure that all coefficients sum up to 1
        self.B /= np.sum(self.B)

        # Update the utility function
        self.update_utility()

    def update_action(self):
        # Update self.action with the action that maximizes the utility function
        self.action = (
            -self.C * self.B[0]
            + self.B[1] * self.pnb
            + self.B[2] * self.ne
            + self.B[3] * self.ee
        ) / (self.B[1] + self.B[2] + self.B[3])

        if self.action > 1:
            self.action = 1
        elif self.action < 0:
            self.action = 0

    def update_empirical(self):
        sum_action_values = 0
        sum_pnb_values = 0
        self.teammates = [
            agent for agent in self.model.schedule.agents if agent != self
        ]  # Exclude the i-th agent
        for agent in self.teammates:
            sum_action_values += agent.action
            sum_pnb_values += agent.pnb
        self.ee = sum_action_values / (len(self.model.schedule.agents) - 1)
        self.ne = sum_pnb_values / (len(self.model.schedule.agents) - 1)

    def step(self):
        self.update_empirical()
        pair_agent = random.choice(self.teammates)
        self.pair_action = pair_agent.action
        self.update_coeffs()
        self.update_action()


class AgentPG(mesa.Agent):
    def __init__(self, unique_id, model, pnb, B, m, update_amount=0.01):
        """
        Initializes the agent for the Public Goods game

        :param unique_id: A unique ID for the agent
        :param model: The current team being runned
        :param pnb: The personal normative belief of the agent ( normal uniform etc. )
        :param B: The initializations of the coefficients of the utility function ( normal constant etc. )
        :param m: The game-specific parameter
        :param update_amount: The learning update amount for RL
        """
        super().__init__(unique_id, model)
        self.action = np.random.normal(loc=0.5)

        if pnb == "normal":
            self.pnb = np.random.normal(loc=0.5)
        elif pnb == "uniform":
            self.pnb = np.random.uniform()

        self.ee = self.pnb
        self.ne = self.pnb

        if B == "normal":
            self.B = np.array(
                [
                    np.random.normal(loc=0.5),
                    np.random.normal(loc=0.5),
                    np.random.normal(loc=0.5),
                    np.random.normal(loc=0.5),
                ]
            )
        elif B == "constant":
            self.B = np.array([0.2, 0.2, 0.2, 0.2])

        self.weight = np.zeros_like(self.B)
        self.m = m
        self.update_amount = update_amount
        self.team_action = np.random.uniform()
        self.teammates = []
        self.update_utility()

    def utility_function(self):
        return (
            self.B[0] * self.weight[0]
            + self.B[1] * self.weight[1]
            + self.B[2] * self.weight[2]
            + self.B[3] * self.weight[3]
        )

    def update_utility(self):
        self.utility = self.utility_function()

    def update_coeffs(self):
        # Compute the weight of the utility function with respect to each coefficient
        self.weight[0] = (1 - self.action) + self.m * self.team_action
        self.weight[1] = -0.5 * (self.action - self.pnb) ** 2
        self.weight[2] = -0.5 * (self.action - self.ne) ** 2
        self.weight[3] = -0.5 * (self.action - self.ee) ** 2

        # Find the index of the coefficient with the largest absolute weight magnitude
        max_index = np.argmax(np.abs(self.weight))

        # Update the selected coefficient slightly
        self.B[max_index] += self.update_amount

        # Ensure that all coefficients sum up to 1
        self.B /= np.sum(self.B)

        # Update the utility function
        self.update_utility()

    def update_action(self):
        # Update self.action with the action that maximizes the utility function
        self.action = (
            self.B[0] * (self.m - 1)
            + self.B[1] * self.pnb
            + self.B[2] * self.ne
            + self.B[3] * self.ee
        ) / (self.B[1] + self.B[2] + self.B[3])

        if self.action > 1:
            self.action = 1
        elif self.action < 0:
            self.action = 0

    def update_empirical(self):
        sum_action_values = 0
        sum_pnb_values = 0
        self.teammates = [
            agent for agent in self.model.schedule.agents if agent != self
        ]  # Exclude the i-th agent
        for agent in self.teammates:
            sum_action_values += agent.action
            sum_pnb_values += agent.pnb
        self.ee = sum_action_values / (len(self.model.schedule.agents) - 1)
        self.ne = sum_pnb_values / (len(self.model.schedule.agents) - 1)

    def step(self):
        self.update_empirical()
        team = random.sample(self.teammates, 3)
        team_actions = [agent.action for agent in team]
        self.team_action = sum(team_actions) + self.action
        self.update_coeffs()
        self.update_action()


class AgentCR(mesa.Agent):
    def __init__(
        self,
        unique_id,
        model,
        pnb,
        B,
        threshold,
        loss_prob,
        update_amount=0.01,
    ):
        """
        Initialize the agent for the Collective Risk game

        :param unique_id: A unique ID for the agent
        :param model: The current team being runned
        :param pnb: The personal normative belief of the agent ( normal uniform etc. )
        :param B: The initializations of the coefficients of the utility function ( normal constant etc. )
        :param threshold: The first game-specific parameter
        :param loss_prob: The second game-specific parameter
        :param update_amount: The learning update amount for RL
        """
        super().__init__(unique_id, model)
        self.action = np.random.normal(loc=0.5)

        if pnb == "normal":
            self.pnb = np.random.normal(loc=0.5)
        elif pnb == "uniform":
            self.pnb = np.random.uniform()

        self.ee = self.pnb
        self.ne = self.pnb

        if B == "normal":
            self.B = np.array(
                [
                    np.random.normal(loc=0.5),
                    np.random.normal(loc=0.5),
                    np.random.normal(loc=0.5),
                    np.random.normal(loc=0.5),
                ]
            )
        elif B == "constant":
            self.B = np.array([0.2, 0.2, 0.2, 0.2])

        self.weight = np.zeros_like(self.B)
        self.update_amount = update_amount
        self.threshold = threshold
        self.loss_prob = loss_prob
        self.team_action = np.random.uniform()
        self.teammates = []
        self.update_utility()

    def utility_function(self):
        return (
            self.B[0] * self.weight[0]
            + self.B[1] * self.weight[1]
            + self.B[2] * self.weight[2]
            + self.B[3] * self.weight[3]
        )

    def update_utility(self):
        self.utility = self.utility_function()

    def update_coeffs(self):
        # Compute the weight of the utility function with respect to each coefficient
        if self.team_action < self.threshold and np.random.random() < self.loss_prob:
            self.weight[0] = 0
        else:
            self.weight[0] = 1 - self.action
        self.weight[1] = -0.5 * (self.action - self.pnb) ** 2
        self.weight[2] = -0.5 * (self.action - self.ne) ** 2
        self.weight[3] = -0.5 * (self.action - self.ee) ** 2

        # Find the index of the coefficient with the largest absolute weight magnitude
        max_index = np.argmax(np.abs(self.weight))

        # Update the selected coefficient slightly
        self.B[max_index] += self.update_amount

        # Ensure that all coefficients sum up to 1
        self.B /= np.sum(self.B)

        # Update the utility function
        self.update_utility()

    def update_action(self):
        # Update self.action with the action that maximizes the utility function
        self.action = (
            self.B[0] * (-1)
            + self.B[1] * self.pnb
            + self.B[2] * self.ne
            + self.B[3] * self.ee
        ) / (self.B[1] + self.B[2] + self.B[3])

        if self.action > 1:
            self.action = 1
        elif self.action < 0:
            self.action = 0

    def update_empirical(self):
        sum_action_values = 0
        sum_pnb_values = 0
        self.teammates = [
            agent for agent in self.model.schedule.agents if agent != self
        ]  # Exclude the i-th agent
        for agent in self.teammates:
            sum_action_values += agent.action
            sum_pnb_values += agent.pnb
        self.ee = sum_action_values / (len(self.model.schedule.agents) - 1)
        self.ne = sum_pnb_values / (len(self.model.schedule.agents) - 1)

    def step(self):
        self.update_empirical()
        team = random.sample(self.teammates, 3)
        team_actions = [agent.action for agent in team]
        self.team_action = sum(team_actions) + self.action
        self.update_coeffs()
        self.update_action()


class Model(mesa.Model):
    def __init__(self, unique_id, game, N, pnb, B, C, p, update_amount=0.01, seed=None):
        """
         Initialize the team
         
         :param unique_id: A unique ID for the team
         :param game: The type of game to use ( Coor, CPD, PG, CR )
         :param N: The number of agents to initialize ( int )
         :param pnb: The personal normative belief of the agent ( normal uniform etc. )
         :param B: Game-specific parameter
         :param C: Game-specific parameter
         :param p: Game-specific parameter
         :param update_amount: The learning update amount for RL
         :param seed: Seed for reproducibility
        """
        super().__init__(unique_id)

        # Set seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_agents = N
        self.schedule = mesa.time.RandomActivation(self)

        # Initialize agents
        for i in range(self.num_agents):
            if game == "Coor":
                a = AgentCoor(i, self, pnb, B, C, update_amount)
            elif game == "CPD":
                a = AgentCPD(i, self, pnb, B, C)
            elif game == "PG":
                a = AgentPG(i, self, pnb, B, C)
            elif game == "CR":
                a = AgentCR(i, self, pnb, B, C, p)
            # Add the agent to the scheduler
            self.schedule.add(a)

    def step(self):
        self.schedule.step()
