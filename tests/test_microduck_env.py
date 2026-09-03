import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from mjlab_microduck.tasks import microduck_env_cfg


def _make_venv(num_envs=2):
    cfg = microduck_env_cfg()
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    return RslRlVecEnvWrapper(env)

def test_obs_and_action_shape():
    venv = _make_venv()
    obs = venv.get_observations()
    assert obs["actor"].shape == (2, 48)
    assert obs["critic"].shape == (2, 48)
    assert venv.action_space.shape == (2, 14)

def test_random_Steps_are_finite():
    venv = _make_venv()
    obs = venv.get_observations()
    for _ in range(20):
        a = 2 * torch.rand(venv.action_space.shape) - 1
        obs, rew, dones, extras = venv.step(a)
        assert torch.isfinite(obs["actor"]).all()
        assert torch.isfinite(rew).all()