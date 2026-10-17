"""Write a fake Adult-shaped CSV for smoke tests (no real data, not for results)."""
import sys
import numpy as np
import pandas as pd

n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
out = sys.argv[1] if len(sys.argv) > 1 else "data/fake_adult.csv"
rng = np.random.RandomState(0)
age = rng.randint(17, 80, n)
edu = rng.randint(1, 17, n)
hours = np.clip(rng.normal(40, 12, n), 1, 99).round()
sex = rng.choice(["Male", "Female"], n, p=[0.67, 0.33])
score = 0.04 * age + 0.3 * edu + 0.03 * hours + 0.5 * (sex == "Male") + rng.normal(0, 1.5, n)
income = np.where(score > np.quantile(score, 0.76), ">50K", "<=50K")
gain = np.where(rng.rand(n) < 0.08, rng.randint(100, 99999, n), 0)
loss = np.where(rng.rand(n) < 0.05, rng.randint(100, 3000, n), 0)
pick = lambda opts, p=None: rng.choice(opts, n, p=p)
df = pd.DataFrame({
    "age": age,
    "workclass": pick(["Private", "Self-emp", "Gov", "?"], [0.7, 0.1, 0.13, 0.07]),
    "fnlwgt": rng.randint(20000, 400000, n),
    "education": pick(["HS-grad", "Bachelors", "Masters"]),
    "education-num": edu,
    "marital-status": pick(["Married", "Never-married", "Divorced", "Widowed"], [.45, .33, .17, .05]),
    "occupation": pick(["Sales", "Craft", "Exec", "Tech", "?", "Other"]),
    "relationship": pick(["Husband", "Wife", "Own-child", "Unmarried"]),
    "race": pick(["White", "Black", "Asian", "Other"], [0.85, 0.1, 0.03, 0.02]),
    "sex": sex,
    "capital-gain": gain,
    "capital-loss": loss,
    "hours-per-week": hours,
    "native-country": pick(["United-States", "Mexico", "India", "?", "Cuba"], [.89, .04, .02, .02, .03]),
    "class": income,
})
df.to_csv(out, index=False)
print(f"wrote {out} ({n} rows, {np.mean(income == '>50K'):.1%} positive)")
