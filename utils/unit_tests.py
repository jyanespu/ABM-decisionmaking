import unittest
from mesa_model import *


class TestAgentCoor(unittest.TestCase):
    def setUp(self):
        self.model = Model(1, "Coor", 5, "normal", "normal", 0.1)
        self.agent = AgentCoor(1, self.model, "normal", "normal", 0.1)

    def test_init(self):
        self.assertEqual(self.agent.unique_id, 1)
        self.assertEqual(self.agent.model, self.model)
        self.assertEqual(self.agent.C, 0.1)
        self.assertEqual(self.agent.update_amount, 0.01)

    def test_utility_function(self):
        self.agent.weight = np.array([0.1, -0.5, -0.5, -0.5])
        expected_utility = (
            self.agent.B[0] * self.agent.weight[0]
            + self.agent.B[1] * self.agent.weight[1]
            + self.agent.B[2] * self.agent.weight[2]
            + self.agent.B[3] * self.agent.weight[3]
        )
        self.assertEqual(self.agent.utility_function(), expected_utility)

    def test_update_utility(self):
        self.agent.weight = np.array([0.1, -0.5, -0.5, -0.5])
        expected_utility = (
            self.agent.B[0] * self.agent.weight[0]
            + self.agent.B[1] * self.agent.weight[1]
            + self.agent.B[2] * self.agent.weight[2]
            + self.agent.B[3] * self.agent.weight[3]
        )
        self.agent.update_utility()
        self.assertEqual(self.agent.utility, expected_utility)

    def test_update_coeffs(self):
        self.agent.action = 0.5
        self.agent.ee = 0.5
        self.agent.pnb = 0.5
        self.agent.ne = 0.5
        self.agent.update_coeffs()
        self.assertAlmostEqual(np.sum(self.agent.B), 1)

    def test_update_action(self):
        self.agent.B = np.array([0.2, 0.2, 0.2, 0.2])
        self.agent.C = 0.1
        self.agent.ee = 0.5
        self.agent.pnb = 0.5
        self.agent.ne = 0.5
        self.agent.update_action()
        self.assertGreaterEqual(self.agent.action, 0)
        self.assertLessEqual(self.agent.action, 1)

    def test_update_empirical(self):
        self.agent.update_empirical()
        self.assertNotEqual(self.agent.ee, 0.5)
        self.assertNotEqual(self.agent.ne, 0.5)

    def test_step(self):
        self.agent.update_empirical = lambda: None
        self.agent.update_coeffs = lambda: None
        self.agent.update_action = lambda: None
        self.agent.step()
        self.assertTrue(
            True
        )  # No assertion, just checking if the step function runs without errors


class TestAgentCPD(unittest.TestCase):
    def setUp(self):
        self.model = Model(1, "CPD", 5, "normal", "normal", 0.1)
        self.agent = AgentCPD(1, self.model, "normal", "normal", 0.1)

    # ... (similar tests for AgentCPD)


class TestAgentPG(unittest.TestCase):
    def setUp(self):
        self.model = Model(1, "PG", 5, "normal", "normal", 0.1)
        self.agent = AgentPG(1, self.model, "normal", "normal", 0.1)

    # ... (similar tests for AgentPG)


class TestAgentCR(unittest.TestCase):
    def setUp(self):
        self.model = Model(1, "CR", 5, "normal", "normal", 0.1, 0.2)
        self.agent = AgentCR(1, self.model, "normal", "normal", 0.1, 0.2)

    # ... (similar tests for AgentCR)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model(1, "Coor", 5, "normal", "normal", 0.1)

    def test_init(self):
        self.assertEqual(self.model.num_agents, 5)
        self.assertIsInstance(self.model.schedule, mesa.time.RandomActivation)

    def test_step(self):
        self.model.schedule.step()
        self.assertTrue(
            True
        )  # No assertion, just checking if the step function runs without errors


if __name__ == "__main__":
    unittest.main()
