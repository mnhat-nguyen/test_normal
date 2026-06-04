import logging
import os
import sys


def get_logger(rank: int, log_dir: str = './logs') -> logging.Logger:
    """
    Return a logger that:
      • writes to stdout  (all ranks)
      • writes to a per-rank log file  logs/rank<N>.log
    Log lines are prefixed with [rank N] for easy grep.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f'rank{rank}')
    logger.setLevel(logging.DEBUG)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter(
        fmt='%(asctime)s [rank %(rankno)s] %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # per-rank file handler
    fh = logging.FileHandler(os.path.join(log_dir, f'rank{rank}.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Inject rank number into every record automatically
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rankno = rank
        return record

    logging.setLogRecordFactory(record_factory)

    return logger
