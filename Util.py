
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def postprocess(episodes, n_runs, rewards, steps, map_size, somatic):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=n_runs),
            "Rewards": rewards.flatten(order="F"),
            "Steps": steps.flatten(order="F"),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")

    if somatic:
        res["map_size"] = np.repeat(f"{map_size}x{map_size} somatic", res.shape[0])
    else:
        res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    if somatic:
        st["map_size"] = np.repeat(f"{map_size}x{map_size} somatic", st.shape[0])
    else:
        st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st


def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    #img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    #fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_states_actions_distribution(states, actions, map_size):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    #img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    #fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()

def plot_steps_and_rewards(rewards_df, steps_df):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()
    #img_title = "frozenlake_steps_and_rewards.png"
    #fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()

def plot_somatic_memory(map_size:int, somaticMemory:dict):
    states = np.arange((map_size * map_size), dtype=int)

    directions =[["←", "↓"],[ "→", "↑"]]

    fig, axs = plt.subplots(nrows=map_size, ncols=map_size, figsize=(15, 15))

    for st in states:
        row = st/map_size
        col = st%map_size

        values = getMarkerActions(st,somaticMemory)

        values = values.reshape(2,2)

        sns.heatmap(
                values,
                annot=directions,
                fmt="",
                ax=axs[int(row),int(col)],
                cmap=sns.diverging_palette(20, 220, as_cmap=True),
                annot_kws={"fontsize": "xx-large"},
                vmin=-1.0,
                vmax=1.0,
                linecolor="black",
                xticklabels=[],
                yticklabels=[],
                cbar= False,
                linewidths=1.0,
            )
        axs[int(row),int(col)].set_title("{}".format(st))
        
    fig.subplots_adjust(wspace=0.1,hspace=0.5)  
    fig.suptitle("Somatic Memory (Actions per States)")
    plt.show()

def getMarkerActions(state: int, somaticMemory: dict):
    ret = np.zeros(4)
    for key in somaticMemory:
        if key[0] == state:
            ret[key[1]] = somaticMemory[key]

    return ret