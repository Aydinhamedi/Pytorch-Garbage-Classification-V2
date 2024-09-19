# Libs >>>
import os
import math
import copy
import torch
import matplotlib.pyplot as plt


# Funcs >>>
def CosineAnnealingLR_Warmup(
    optimizer,
    warmup_iters,
    main_iters,
    lr_idling_iters,
    decay_iters,
    lr_main_min,
    lr_final_min,
    warmup_start=0.05,
    warmup_type="exponential",
):
    """
    Creates a learning rate scheduler that combines a warmup phase, a cosine annealing phase, and a linear decay phase.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        warmup_iters (int): The number of iterations for the warmup phase.
        main_iters (int): The number of iterations for the main cosine annealing phase.
        decay_iters (int): The number of iterations for the linear decay phase.
        lr_main_min (float): The minimum learning rate after the cosine annealing phase.
        lr_final_min (float): The final minimum learning rate after the linear decay phase.
        warmup_start (float, optional): The starting factor for the warmup phase. Default is 0.008.
        warmup_type (str, optional): "linear" or "exponential". Default is "exponential".
    Returns:
        torch.optim.lr_scheduler.SequentialLR: A combined learning rate scheduler with warmup, cosine annealing, and linear decay phases.

    """
    # Warmup phase
    match warmup_type:
        case "linear":
            lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: warmup_start
                + (1 - warmup_start) * min(step / warmup_iters, 1.0),
            )  # I didnt used LinearLR because it was buggy
        case "exponential":
            lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: (
                    (optimizer.param_groups[0]["lr"] * warmup_start)
                    * (
                        optimizer.param_groups[0]["lr"]
                        / (optimizer.param_groups[0]["lr"] * warmup_start)
                    )
                    ** min(step / warmup_iters, 1)
                )
                / optimizer.param_groups[0]["lr"],
            )
        case _:
            raise NotImplementedError
    # Linear decay phase
    lr_scheduler_final = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=lr_main_min / optimizer.param_groups[0]["initial_lr"],
        end_factor=lr_final_min / optimizer.param_groups[0]["initial_lr"],
        total_iters=decay_iters,
    )
    # Cosine annealing phase
    lr_scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=main_iters, eta_min=lr_main_min
    )
    # Merging the three schedulers
    CosineAnnealingLR_Warmup_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [lr_scheduler_warmup, lr_scheduler_main, lr_scheduler_final],
        milestones=[
            warmup_iters + lr_idling_iters,
            warmup_iters + main_iters + lr_idling_iters,
        ],
    )
    CosineAnnealingLR_Warmup_lr_scheduler.__class__.__name__ = (
        "CosineAnnealing_pWarmup&Decay"
    )
    return CosineAnnealingLR_Warmup_lr_scheduler


def Profile(
    lr_scheduler,
    iter_max,
    monitor=["lr"],
    show_plot=False,
    save_plot=True,
    save_path="Debug/lr_scheduler_profile",
):
    """
    Profiles the learning rate scheduler over a specified number of iterations.

    Args:
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to profile.
        iter_max (int): The maximum number of iterations to profile.
        monitor (list of str): List of parameters to monitor. Default is ["lr"].
        show_plot (bool): Whether to display the plot. Default is False.
        save_plot (bool): Whether to save the plot. Default is True.
        save_path (str): The path to save the plot. Default is "Debug/lr_scheduler_profile.png".

    Returns:
        dict: A dictionary containing the monitored parameters over iterations.
    """
    # Clone the learning rate scheduler to avoid modifying the original one
    cloned_scheduler = copy.deepcopy(lr_scheduler)

    profile_data = {key: [] for key in monitor}

    for _ in range(iter_max):
        cloned_scheduler.step()
        for key in monitor:
            if key == "lr":
                profile_data[key].append(cloned_scheduler.get_last_lr())
            elif key in cloned_scheduler.optimizer.defaults:
                profile_data[key].append(cloned_scheduler.optimizer.state_dict()["param_groups"][0][key])

    if show_plot or save_plot:
        num_plots = len(monitor)
        _, axs = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
        if num_plots == 1:
            axs = [axs]
        for i, key in enumerate(monitor):
            axs[i].plot(profile_data[key], label=key)
            axs[i].set_xlabel("Iteration")
            axs[i].set_ylabel(key)
            axs[i].set_title(f"{key} over Iterations")
            axs[i].legend()
        plt.tight_layout()

        if save_plot:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            scheduler_name = lr_scheduler.__class__.__name__
            save_path = os.path.join(
                os.path.dirname(save_path), f"{scheduler_name}_profile.png"
            )
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        else:
            plt.close()

    return profile_data


# Optional Profile (You can use this to profile and tune the scheduler)
Do_Profile = True
if __name__ == "__main__" and Do_Profile:
    # Define the optimizer
    optimizer = torch.optim.SGD(
        [torch.randn(1)],
        lr=0.01,
        weight_decay=0.0005,
        momentum=0.9,
        # nesterov=True,
    )
    # Define the learning rate scheduler to be profiled
    lr_scheduler = CosineAnnealingLR_Warmup(  # Very similar to the onecycleLR
        optimizer,
        warmup_iters=10,
        main_iters=60,
        lr_idling_iters=12,
        decay_iters=146,
        lr_main_min=0.006,  # 0.006
        lr_final_min=0.004,  # 0.002
        warmup_start=0.07,
    )
    
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        three_phase=True,
        cycle_momentum=True,
        max_lr=0.0075,
        epochs=256,
        pct_start=0.15,
        final_div_factor=50,
        steps_per_epoch=1,
    )
    # Profile the learning rate scheduler
    Profile(lr_scheduler, 256, show_plot=True, save_plot=False, monitor=["lr", "momentum"])
