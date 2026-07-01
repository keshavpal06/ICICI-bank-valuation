import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data = {
    'Bank': ['ICICI Bank', 'Kotak Mahindra', 'Axis Bank', 'SBI', 'IndusInd Bank', 'HDFC Bank'],
    'ROE': [16.10, 11.20, 13.10, 15.40, 1.36, 13.80],
    'PB': [2.73, 2.17, 1.95, 1.59, 1.11, 2.12]
}

df = pd.DataFrame(data)

# Regression excluding IndusInd (distressed outlier)
df_clean = df[df['Bank'] != 'IndusInd Bank']
slope, intercept, r, p, se = stats.linregress(df_clean['ROE'], df_clean['PB'])

x_line = np.linspace(df_clean['ROE'].min() - 1, df_clean['ROE'].max() + 1, 100)
y_line = slope * x_line + intercept

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for _, row in df.iterrows():
    color = 'red' if row['Bank'] == 'IndusInd Bank' else 'steelblue'
    ax.scatter(row['ROE'], row['PB'], color=color, s=100, zorder=5)
    ax.annotate(row['Bank'], (row['ROE'], row['PB']),
                textcoords="offset points", xytext=(8, 4), fontsize=9)

ax.plot(x_line, y_line, 'k--', linewidth=1.5,
        label=f'Regression line (R²={r**2:.2f})')

# ICICI predicted vs actual
icici = df[df['Bank'] == 'ICICI Bank'].iloc[0]
predicted_pb = slope * icici['ROE'] + intercept
premium = ((icici['PB'] - predicted_pb) / predicted_pb) * 100

ax.annotate(f"ICICI: Actual {icici['PB']:.2f}x\nPredicted {predicted_pb:.2f}x\nPremium: {premium:.1f}%",
            xy=(icici['ROE'], icici['PB']),
            xytext=(icici['ROE'] - 3, icici['PB'] + 0.2),
            fontsize=9, color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred'))

ax.set_xlabel('Return on Equity (ROE %)', fontsize=11)
ax.set_ylabel('Price-to-Book (P/B)', fontsize=11)
ax.set_title('ROE vs P/B Regression — Indian Banking Sector\n(IndusInd excluded as distressed outlier)',
             fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roe_pb_regression.png', dpi=150)
plt.show()

print(f"Regression: P/B = {slope:.3f} × ROE + {intercept:.3f}")
print(f"R² = {r**2:.3f}")
print(f"ICICI predicted P/B: {predicted_pb:.2f}x")
print(f"ICICI actual P/B: {icici['PB']:.2f}x")
print(f"Premium to regression line: {premium:.1f}%")