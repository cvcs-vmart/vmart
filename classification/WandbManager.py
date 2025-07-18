import wandb


class WandbManager:
    def __init__(self, project, entity, name, train_params):
        wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=train_params
        )

    def save(self, path):
        if wandb.run is not None:
            wandb.save(path, policy="now")  # policy="now" forza il salvataggio

    def log(self, log_dict):
        if wandb.run is not None:
            wandb.log(log_dict)
