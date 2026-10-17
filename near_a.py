import yaml
from src.data import load_adult, split_data
from src.preprocess import Preprocessor
from src.evaluate import PrivacyEvaluator
spec = yaml.safe_load(open("config/adult.yaml"))
pc = yaml.safe_load(open("config/privacy.yaml"))
cfg = yaml.safe_load(open("config/experiment.yaml"))
df = load_adult(spec, "data/adult.csv")
tr, va, te = split_data(df, spec["target"], cfg["split_seed"])
pre = Preprocessor(spec).fit(tr)
p = PrivacyEvaluator(pre, pre.group(tr), pre.group(va), pc)
d = p.dA
print("half-A near-share:", ((d > 0) & (d < p.threshold)).sum() / (d > 0).sum())
