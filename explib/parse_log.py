import re
from pathlib import Path

from DejaVu.config import DejaVuConfig


def parse_log(config: DejaVuConfig):
    log_path = Path(config.output_dir / "log")
    total_re = r'\|/train_exp_CFL\s*\|\s*100\.00\%\|\s*100\.00\%' \
               r'\|\s*1\|\s*(?P<total>[0-9\.]+)s\|\s*[0-9\.]+\(±\s*[0-9\.]+\)s\|\s*[0-9\.]+~\s*[0-9\.]+\|'
    start_test_re = r'=+\s*Start\s*Test at Epoch \d+ =+'
    end_test_re = r'=+\s*End\s*Test at Epoch \d+ =+'
    epoch_re = r'.*callbacks:on_validation_epoch_end.* - epoch=(?P<epoch>\d+)'
    valid_epoch_re = r'.*trainer.checkpoint_callback.best_model_path=' \
                     r'\'.*/lightning_logs/version_0/checkpoints/' \
                     r'epoch=(?P<valid_epoch>\d+)' \
                     r'-A@1=(?P<A1>[0-9.]+)' \
                     r'-val_loss=(?P<val_loss>[0-9.]+)' \
                     r'-MAR=(?P<MAR>[0-9.]+).ckpt.*'
    within_test = False
    with open(log_path) as f:
        lines = f.readlines()
        for line in lines:
            if match := re.match(total_re, line):
                total = float(match.group('total'))
                continue
            if match := re.match(epoch_re, line):
                epoch = int(match.group('epoch')) + 1
                continue
            if match := re.match(valid_epoch_re, line):
                valid_epoch = int(match.group('valid_epoch'))
                val_loss = float(match.group('val_loss'))
                val_MAR = float(match.group('MAR'))
                val_A1 = float(match.group('A1'))
                continue
            if re.match(start_test_re, line):
                within_test = True
                continue
            if re.match(end_test_re, line):
                within_test = False
                continue
            if within_test:
                if match := re.match(
                        r'A@1=\s*(?P<A1>[0-9.]+)\s*% A@2=\s*(?P<A2>[0-9.]+)\s*% A@3=\s*(?P<A3>[0-9.]+)\s*% '
                        r'A@5=\s*(?P<A5>[0-9.]+)\s*% MAR=\s*(?P<MAR>[0-9.]+)\s*',
                        line):
                    metrics = {
                        'A@1': float(match.group('A1')),
                        'A@2': float(match.group('A2')),
                        'A@3': float(match.group('A3')),
                        'A@5': float(match.group('A5')),
                        'MAR': float(match.group('MAR')),
                    }
    return f"{metrics['A@1']},{metrics['A@2']},{metrics['A@3']},{metrics['A@5']}," \
           f"{metrics['MAR']},{total},{epoch},{valid_epoch},{log_path!s}," \
           f"{val_loss},{val_MAR},{val_A1}," \
           f"{config.get_reproducibility_info()['command_line']}," \
           f"{config.get_reproducibility_info().get('git_url', '')},"
