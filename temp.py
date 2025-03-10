from omegaconf import OmegaConf

a = OmegaConf.load("conf.yaml")
print(**a.model.get("params"))