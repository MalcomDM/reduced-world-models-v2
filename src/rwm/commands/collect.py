
import typer
from pathlib import Path
from typing import Dict, Callable

from rwm.data.collector import collect_rollout
from rwm.types import PolicyName
from rwm.policies.base_policy import BasePolicy
from rwm.policies.random_policy import RandomPolicy
from rwm.policies.human_policy import HumanPolicy


app = typer.Typer()

POLICIES: Dict[PolicyName, Callable[[], BasePolicy]] = {
    PolicyName.RANDOM: 			lambda: RandomPolicy(smooth=False),
    PolicyName.RANDOM_SMOOTH: 	lambda: RandomPolicy(smooth=True),
    PolicyName.HUMAN: 			lambda: HumanPolicy()
}


@app.command()
def run(
    env_name:		str = typer.Option( "car_racing", help="Name of the gym environment" ),
    policy:	 PolicyName = typer.Option( "random", help="Policy name" ),
    scenario: 		str = typer.Option( "basic_turn", help="Scenario identifier" ),
    render_mode: 	str = typer.Option( "rgb_array", help="rgb_array or human" ),
    out_dir: 		Path= typer.Option( Path("data/rollouts"), help="Base output directory" ),
    max_steps: 		int = typer.Option( 300, help="Maximum number of steps per rollout" ),
    idle_threshold: int = typer.Option( 20, help="Steps with no reward before ending rollout early" ),
):
	if policy not in POLICIES:
		typer.echo(f"❌ Unknown policy '{policy}'. Available: {list(POLICIES.keys())}")
		raise typer.Exit(code=1)

	if render_mode=="human":
		policy_instance = POLICIES[PolicyName.HUMAN]()
	else:
		policy_instance = POLICIES[policy]()
		policy_instance.reset()

	collect_rollout(
		env_name=env_name,
		policy_fn=policy_instance.act,
		scenario_name=scenario,
		out_dir=out_dir,
		max_steps=max_steps,
		idle_threshold=idle_threshold,
		render_mode=render_mode,
	)


@app.command()
def bulk(
	env_name: str = typer.Option("car_racing"),
	scenario: str = typer.Option("base_rollouts"),
	render_mode: str = typer.Option("rgb_array"),
	out_dir: Path = typer.Option(Path("data/rollouts")),
	max_steps: int = typer.Option(1000),
	idle_threshold: int = typer.Option(100),
	early_push: int = typer.Option(20),
	num_each: int = typer.Option(10),
):
	""" Run multiple rollouts: [num_each] for each policy type (random, smooth). """
	for policy_name in [PolicyName.RANDOM, PolicyName.RANDOM_SMOOTH]:
		typer.echo(f"\n▶ Generating {num_each} rollouts for policy: {policy_name.value}")
		for _ in range(num_each):
			policy_instance = POLICIES[policy_name]()
			policy_instance.reset()

			tagged_scenario = f"{scenario}_{policy_name.value}"
			collect_rollout(
				env_name=env_name,
				policy_fn=policy_instance.act,
				scenario_name=tagged_scenario,
				out_dir=out_dir,
				max_steps=max_steps,
				idle_threshold=idle_threshold,
				render_mode=render_mode,
				early_push=early_push,
			)

if __name__ == "__main__":
    app()
