import os
import argparse
import numpy as np

from utils import Model
from utils import get_args, plots

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="MESA model")
    args = get_args(parser)

    # Simulation begins
    num_groups = args.groups
    num_agents_per_group = args.group_size
    num_rounds = args.rounds

    all_coefficients = []
    all_indiv_actions = []
    all_indiv_actions_last = []
    all_pnb = []
    group_actions = np.zeros((num_rounds, num_agents_per_group))

    for j in range(num_groups):
        model = Model(
            j,
            args.game,
            num_agents_per_group,
            args.pnb,
            args.B,
            args.G1,
            args.G2,
            args.update_amount,
            seed=j,
        )
        for i in range(num_rounds):
            model.step()

            # Store the results
            agent_num = 0
            for agent in model.schedule.agents:
                if i == (num_rounds - 1):  # If last round append coefficients
                    all_coefficients.append(agent.B)
                    all_indiv_actions_last.append(agent.action)
                    all_pnb.append(agent.pnb)
                group_actions[i, agent_num] = agent.action
                all_indiv_actions.append(agent.action)
                agent_num += 1

        mean_group_actions = np.mean(
            group_actions, axis=1
        )  # Get mean of current group actions
        # Add new row to all_group_actions representing the mean by round in the current group
        if j == 0:
            all_group_actions = mean_group_actions
        else:
            all_group_actions = np.vstack((all_group_actions, mean_group_actions))

    # Create a directory to save the pictures
    if args.game == "CR":
        output_directory = f"{args.directory}/{args.game}_{args.groups}_{args.group_size}_{args.rounds}_{args.pnb}_{args.B}_{args.G1}_{args.G2}"
    else:
        output_directory = f"{args.directory}/{args.game}_{args.groups}_{args.group_size}_{args.rounds}_{args.pnb}_{args.B}_{args.G1}_{args.update_amount}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Visualize the results
    plots(
        output_directory,
        all_coefficients,
        all_indiv_actions,
        all_indiv_actions_last,
        all_group_actions,
        all_pnb,
    )
