import matplotlib.pyplot as plt
import numpy as np
import os

EE_NAME_MAPPING_JOINT_IMPEDANCE = {
    0: "j0",
    1: "j1",
    2: "j2",
    3: "j3",
    4: "j4",
    5: "j5",
    6: "j6",
    7: "f0",
    8: "f1",
    9: "f2",
    10: "f3",
    11: "f4",
    12: "f5",
    13: "f6",
    14: "f7",
    15: "f8",
    16: "f9",
    17: "f10",
    18: "f11",
    19: "f12",
    20: "f13",
    21: "f14",
    22: "f15",
}

EE_NAME_MAPPING_OSC_POSE = {
    0: "x",
    1: "y",
    2: "z",
    3: "r1",
    4: "r2",
    5: "r3",
    6: "r4",
    7: "r5",
    8: "r6",
    9: "f0",
    10: "f1",
    11: "f2",
    12: "f3",
    13: "f4",
    14: "f5",
    15: "f6",
    16: "f7",
    17: "f8",
    18: "f9",
    19: "f10",
    20: "f11",
    21: "f12",
    22: "f13",
    23: "f14",
    24: "f15"
}

def plot_image(gt, name, pred=None, step_log=None, wandb=None, folder_name="figures"):
    time_i = [i for i in range(gt.shape[0])]
    # Create a figure and a grid of subplots
    num_sub_plot = gt.shape[-1]
    # figure size is iin (width, height)
    fig, axes = plt.subplots(
        num_sub_plot, figsize=(6, 40), height_ratios=[1] * num_sub_plot
    )  # 7 rows, 1 columns
    EE_NAME_MAPPING = EE_NAME_MAPPING_JOINT_IMPEDANCE if num_sub_plot == 23 else EE_NAME_MAPPING_OSC_POSE
    # Display each image in a subplot
    for i in range(num_sub_plot):
        axes[i].plot(time_i, gt[:, i].tolist(), color="green", label="gt")
        if pred is not None:
            axes[i].plot(time_i, pred[:, i].tolist(), color="red", label="pred")
        axes[i].set_title(f"{EE_NAME_MAPPING[i]}")

    # Show legend
    axes[num_sub_plot - 1].legend()
    plt.tight_layout()
    # fig.set_size_inches(18.5, 10.5, forward=True)
    os.makedirs(folder_name, exist_ok=True)

    plt.savefig(f"{folder_name}/{name}_pred_vs_gt.jpg")

    # Log to wandb
    if step_log:
        step_log.update({f"{name}_pred_vs_gt": wandb.Image(fig)})
        return step_log
