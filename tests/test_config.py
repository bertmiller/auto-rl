from pathlib import Path
from auto_rl.config import ProblemConfig, load_config

FIXTURE = Path(__file__).parent / "fixtures" / "sample_problem" / "problem.toml"

def test_load_config():
    config = load_config(FIXTURE)
    assert config.name == "sample-maximize"
    assert config.metric_direction == "maximize"
    assert config.oracle_command == "python3 verify.py"
    assert "solution.py" in config.editable_files
    assert "verify.py" in config.readable_files
    assert config.reward_type == "log"
    assert config.max_attempts == 10
    assert config.challenge_dir.exists()

def test_config_defaults():
    config = load_config(FIXTURE)
    assert config.sandbox_pool_size == 2
    assert config.oracle_timeout == 30

def test_config_system_prompt_template():
    config = load_config(FIXTURE)
    rendered = config.render_system_prompt(max_attempts=10, baseline_metric=1.0)
    assert "10 attempts" in rendered
