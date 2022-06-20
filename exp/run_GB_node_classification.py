from sklearn.ensemble import GradientBoostingClassifier

from DejaVu.config import DejaVuConfig
from DejaVu.workflow import train_exp_sklearn_classifier
from DejaVu.models.get_model import ClassifierProtocol
from failure_dependency_graph import FDG


def get_model(cdp: FDG, config: DejaVuConfig) -> ClassifierProtocol:
    return GradientBoostingClassifier()


if __name__ == '__main__':
    train_exp_sklearn_classifier(DejaVuConfig().parse_args(), get_model)
