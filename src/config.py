import os
import configparser
from dataclasses import dataclass

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".config"))

@dataclass
class ModelConfig:
    ntoken: int = 9714
    ninp: int = 512
    nhead: int = 8
    n_kv_heads: int = 2
    nhid: int = 2048
    nlayers: int = 9
    dropout: float = 0.1
    max_seq_len: int = 256

@dataclass
class TrainConfig:
    batch_size: int = 8
    grad_accum_steps: int = 2
    stride: int = 128
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    epochs: int = 2
    eval_interval: int = 50
    save_interval: int = 250

def load_config(config_path=CONFIG_PATH):
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if os.path.exists(config_path):
        parser = configparser.ConfigParser()
        parser.read(config_path)

        if "model" in parser:
            m = parser["model"]
            if "ntoken" in m:
                model_cfg.ntoken = int(m["ntoken"])
            if "ninp" in m:
                model_cfg.ninp = int(m["ninp"])
            if "nhead" in m:
                model_cfg.nhead = int(m["nhead"])
            if "n_kv_heads" in m:
                model_cfg.n_kv_heads = int(m["n_kv_heads"])
            if "nhid" in m:
                model_cfg.nhid = int(m["nhid"])
            if "nlayers" in m:
                model_cfg.nlayers = int(m["nlayers"])
            if "dropout" in m:
                model_cfg.dropout = float(m["dropout"])
            if "max_seq_len" in m:
                model_cfg.max_seq_len = int(m["max_seq_len"])

        if "training" in parser:
            t = parser["training"]
            if "batch_size" in t:
                train_cfg.batch_size = int(t["batch_size"])
            if "grad_accum_steps" in t:
                train_cfg.grad_accum_steps = int(t["grad_accum_steps"])
            if "stride" in t:
                train_cfg.stride = int(t["stride"])
            if "learning_rate" in t:
                train_cfg.learning_rate = float(t["learning_rate"])
            if "min_lr" in t:
                train_cfg.min_lr = float(t["min_lr"])
            if "weight_decay" in t:
                train_cfg.weight_decay = float(t["weight_decay"])
            if "grad_clip" in t:
                train_cfg.grad_clip = float(t["grad_clip"])
            if "epochs" in t:
                train_cfg.epochs = int(t["epochs"])
            if "eval_interval" in t:
                train_cfg.eval_interval = int(t["eval_interval"])
            if "save_interval" in t:
                train_cfg.save_interval = int(t["save_interval"])

    return model_cfg, train_cfg

default_model_config, default_train_config = load_config()
