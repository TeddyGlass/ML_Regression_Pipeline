import optuna


class Optimizer:

    def __init__(self, obj):
        self.obj = obj

    def search(self):
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(seed=1541)
        )
        study.optimize(self.obj, n_trials=60, n_jobs=-1)
