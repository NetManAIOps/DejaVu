from DejaVu.config import DejaVuConfig
from DejaVu.models import get_GAT_model
from DejaVu.workflow import train_exp_CFL

if __name__ == '__main__':
    # logger.info("Disable JIT because of DGL")
    # set_jit_enabled(False)
    train_exp_CFL(DejaVuConfig().parse_args(), get_GAT_model, plot_model=False)
