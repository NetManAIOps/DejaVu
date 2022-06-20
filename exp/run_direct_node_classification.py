from DejaVu.config import DejaVuConfig
from DejaVu.models.get_model import get_DNN_model
from DejaVu.workflow import train_exp_CFL

if __name__ == '__main__':
    train_exp_CFL(DejaVuConfig().parse_args(), get_DNN_model, plot_model=False)
