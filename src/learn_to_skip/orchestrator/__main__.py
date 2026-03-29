"""CLI entry point for the orchestrator."""
import click

from learn_to_skip.orchestrator.runner import ExperimentRunner, EXPERIMENT_ORDER


@click.group()
def cli():
    """LearnToSkip experiment orchestrator."""
    pass


@cli.command("run-all")
@click.option("--datasets", "-d", multiple=True, default=["sift10k"], help="Datasets to use")
@click.option("--force", is_flag=True, help="Re-run completed experiments")
def run_all(datasets, force):
    """Run all experiments in dependency order."""
    runner = ExperimentRunner(datasets=list(datasets), force=force)
    runner.run_all()


@cli.command("run")
@click.argument("experiment_name")
@click.option("--datasets", "-d", multiple=True, default=["sift10k"])
@click.option("--force", is_flag=True)
def run(experiment_name, datasets, force):
    """Run a single experiment."""
    runner = ExperimentRunner(datasets=list(datasets), force=force)
    runner.run_experiment(experiment_name)


@cli.command("plot-all")
def plot_all():
    """Generate all plots from existing results."""
    runner = ExperimentRunner()
    runner.plot_all()


@cli.command("plot")
@click.argument("figure_name")
def plot(figure_name):
    """Generate a single plot."""
    runner = ExperimentRunner()
    # Map figure names to plot functions
    import pandas as pd
    from learn_to_skip.config import RESULTS_DIR
    from learn_to_skip.visualization.plots import (
        plot_waste_ratio_bar, plot_speedup_bar, plot_pareto_scatter,
        plot_roc_curves, plot_threshold_sensitivity,
        plot_scalability_line, plot_recall_over_time,
    )

    plots = {
        "fig1": lambda: plot_waste_ratio_bar(
            pd.read_csv(RESULTS_DIR / "motivation" / "fig1_data.csv"),
            str(RESULTS_DIR / "motivation")),
        "fig2": lambda: plot_speedup_bar(
            pd.read_csv(RESULTS_DIR / "build_speed" / "table2.csv"),
            str(RESULTS_DIR / "build_speed")),
        "fig3": lambda: plot_pareto_scatter(
            pd.read_csv(RESULTS_DIR / "recall" / "pareto_data.csv"),
            str(RESULTS_DIR / "recall")),
        "fig4": lambda: plot_roc_curves(
            RESULTS_DIR / "classifier" / "roc_data.json",
            str(RESULTS_DIR / "classifier")),
        "fig5": lambda: plot_threshold_sensitivity(
            pd.read_csv(RESULTS_DIR / "threshold" / "fig5_data.csv"),
            str(RESULTS_DIR / "threshold")),
        "fig6": lambda: plot_scalability_line(
            pd.read_csv(RESULTS_DIR / "scalability" / "fig6_data.csv"),
            str(RESULTS_DIR / "scalability")),
        "fig7": lambda: plot_recall_over_time(
            pd.read_csv(RESULTS_DIR / "adaptive" / "fig7_data.csv"),
            str(RESULTS_DIR / "adaptive")),
    }

    if figure_name not in plots:
        click.echo(f"Unknown figure: {figure_name}. Available: {list(plots)}")
        return
    plots[figure_name]()


@cli.command("status")
def status():
    """Show experiment completion status."""
    runner = ExperimentRunner()
    st = runner.status()
    click.echo(f"\n{'Experiment':<20} {'Complete':>10} {'Deps Met':>10} {'Ready':>8}")
    click.echo("-" * 52)
    for name in EXPERIMENT_ORDER:
        info = st[name]
        click.echo(
            f"{name:<20} {'YES' if info['complete'] else 'no':>10} "
            f"{'YES' if info['deps_met'] else 'no':>10} "
            f"{'YES' if info['ready'] else '-':>8}"
        )


if __name__ == "__main__":
    cli()
