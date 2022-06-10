import pandas as pd
import seaborn as sns

from config.createconfig import load_config
cfg = load_config()


def plot_perf(trainer):
    file_name = cfg[1]['exp_params']['METRICS_DIR']

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/{file_name}.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    sns.relplot(data=metrics, kind="line")