from imldiff.models import Model


class SHAPExplainer(Model):
    @property
    def _prediction_str(self):
        return f'E_{self}'

    def __call__(self, X):
        raise NotImplementedError()


