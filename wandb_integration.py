"""
Weights & Biases integration helpers for experiment tracking.

Usage: each experiment function calls `init_run(...)` at setup and
`finish_run()` at teardown. Metrics are logged directly via `wandb.log`
in the training loops.

If wandb is not installed or not logged in, all functions become no-ops
so experiments still run without it.
"""

import wandb

_enabled = True


def init_run(experiment_name, config, location, **kwargs):
    """Initialize a W&B run. `config` is the experiment's results dict."""
    global _enabled
    _cleanup_jupyter_hooks()
    try:
        wandb.init(
            project="AxiomaticDeepVoting",
            name=location.split("/")[-1],
            group=experiment_name,
            config=config,
            tags=[
                config.get("architecture", "unknown"),
                config.get("election_sampling", {}).get("probmodel", "unknown"),
            ],
            **kwargs,
        )
        _enabled = True
    except Exception as e:
        print(f"W&B init failed ({e}), continuing without tracking.")
        _enabled = False


def log(metrics, step=None):
    """Log metrics if W&B is active."""
    if _enabled and wandb.run is not None:
        wandb.log(metrics, step=step)


def log_summary(metrics):
    """Write to run summary (final metrics, not time-series)."""
    if _enabled and wandb.run is not None:
        for k, v in metrics.items():
            wandb.run.summary[k] = v


def finish_run():
    """Finish the current W&B run."""
    if _enabled and wandb.run is not None:
        wandb.finish()
        _cleanup_jupyter_hooks()


def _cleanup_jupyter_hooks():
    """Remove wandb's Jupyter cell hooks that outlive the run."""
    try:
        ip = get_ipython()
    except NameError:
        return
    for event in ("pre_run_cell", "post_run_cell"):
        cbs = ip.events.callbacks.get(event, [])
        ip.events.callbacks[event] = [
            cb for cb in cbs
            if "_WandbInit" not in str(getattr(cb, "__qualname__", ""))
        ]
