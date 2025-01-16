import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pyrootutils

# project root setup
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@hydra.main(version_base=None, config_path='config', config_name='run')
def run(cfg: DictConfig) -> None:
    print(f'Running {cfg.model.model_name} on {cfg.task.task_name}...')
    task = instantiate(cfg.task)
    model = instantiate(cfg.model, task=task)
    model.run()

if __name__ == '__main__':
    run()