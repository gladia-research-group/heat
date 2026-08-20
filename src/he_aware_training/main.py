import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from he_aware_training import PROJECT_ROOT


def run(cfg: DictConfig):
    pass


@hydra.main(
    version_base=None,
    config_path=f"{PROJECT_ROOT}/configs",
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    wandb.init(
        entity=cfg.core.entity,
        project=cfg.core.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.core.tags,
    )
    run(cfg=cfg)
    wandb.finish()


if __name__ == "__main__":
    main()
