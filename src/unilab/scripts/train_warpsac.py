"""WarpSAC training/playback entrypoint (shared off-policy implementation)."""

import hydra
from omegaconf import DictConfig

from unilab.scripts.train_offpolicy import main as _offpolicy_main


@hydra.main(version_base="1.3", config_path="../conf/warpsac", config_name="config")
def main(cfg: DictConfig) -> None:
    _offpolicy_main(cfg)


if __name__ == "__main__":
    main()
