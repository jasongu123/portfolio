import pandas as pd
import numpy as np

returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)

# Compute correlation matrix
corr_matrix = returns.corr()

# Get upper triangle (excluding diagonal)
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Average pairwise correlation
avg_corr = upper_triangle.stack().mean()
print(f"Average pairwise correlation: {avg_corr:.4f}")