"""TabM training schedule used by tune_tabm2.py, shared with probe_nn.py (importing tune_tabm2 would run it)."""
SCHEDULE = {"max_epochs": 80, "patience": 10, "lr_decay_patience": 3}
