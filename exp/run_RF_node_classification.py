from DejaVu.config import DejaVuConfig

from DejaVu.models.get_model import get_RF_model
from DejaVu.workflow import train_exp_sklearn_classifier

if __name__ == '__main__':
    train_exp_sklearn_classifier(DejaVuConfig().parse_args(), get_RF_model)
