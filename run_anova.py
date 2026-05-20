import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
import platform

# Add R bin directory to DLL search path on Windows before importing rpy2
if platform.system() == 'Windows':
    r_bin = r"C:\Program Files\R\R-4.6.0\bin\x64"
    if os.path.exists(r_bin):
        os.add_dll_directory(r_bin)
        os.environ['PATH'] = r_bin + os.path.pathsep + os.environ.get('PATH', '')

import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# 1. (pandas-to-R conversion is now handled locally using localconverter context)

# 2. Import R packages
artool = importr('ARTool')
base = importr('base')

# 3. Generate a Statistically Viable Dummy Dataset
np.random.seed(42)

# Simulate 80 distinct phenomenological observations 
# (10 trials for each of the 4 Modes x 2 Scenarios combinations)
observations = [f"Obs_{i}" for i in range(1, 81)]
modes = np.repeat(['Mode1', 'Mode2', 'Mode3', 'Mode4'], 20)
scenarios = np.tile(np.repeat(['Scen1', 'Scen2'], 10), 4)

base_df = pd.DataFrame({
    'Observation': observations, 
    'Mode': modes, 
    'Scenario': scenarios
})

# Both algorithms process the SAME observations (Creating the paired nature)
df_algA = base_df.copy()
df_algA['Algorithm'] = 'AlgA'
df_algA['Error'] = np.random.gamma(shape=2.0, scale=1.5, size=80) 

df_algB = base_df.copy()
df_algB['Algorithm'] = 'AlgB'
# We add +1.5 so AlgB has a mathematically different error for the ANOVA to detect
df_algB['Error'] = np.random.gamma(shape=2.5, scale=1.5, size=80) + 1.5 

df = pd.concat([df_algA, df_algB], ignore_index=True)

# Simulate the validation step failures (Randomly drop 8 samples)
drop_indices = np.random.choice(df.index, size=8, replace=False)
df = df.drop(drop_indices).reset_index(drop=True)

# 4. CRITICAL: Enforce Categorical Types
df['Mode'] = df['Mode'].astype('category')
df['Algorithm'] = df['Algorithm'].astype('category')
df['Scenario'] = df['Scenario'].astype('category')
df['Observation'] = df['Observation'].astype('category')

# 5. Define the Mixed Model Formula
# The random effect (1|Observation) links the Algorithm A and B outputs together
formula_str = "Error ~ Mode * Algorithm * Scenario + (1|Observation)"

# 6. Push to R and Run ART
print("Transferring data to R and calculating Aligned Rank Transform...\n")
with localconverter(ro.default_converter + pandas2ri.converter):
    r_df = ro.conversion.py2rpy(df)

# Fit the ART model
art_model = artool.art(ro.Formula(formula_str), data=r_df)

# Run the ANOVA on the transformed data
anova_results = artool.anova_art(art_model)

# 7. Print the Results
print("--- ANOVA Results (p-values) ---")
print(anova_results)


print("--- Running ART-C Post-Hoc Contrasts ---")
# 2. Run the contrasts for the specific interaction 
# Replace "Mode:Algorithm" with whichever interaction was significant in your ANOVA
# interact=True is CRITICAL for answering "Which specific combination is best?"
contrasts_r = artool.art_con(art_model, "Mode:Algorithm", interact=True)

# 3. Print the formatted summary directly from R
print(base.summary(contrasts_r))

# 4. Optional: Convert the results back into a Python pandas DataFrame 
# so you can filter or export them easily.
# R's art_con returns an emmGrid object. We must coerce it to a DataFrame in R first.
contrasts_df_r = base.as_data_frame(contrasts_r)

# Convert to pandas DataFrame
contrasts_df_py = pandas2ri.rpy2py(contrasts_df_r)

print("\n--- Contrasts as a Pandas DataFrame ---")
print(contrasts_df_py.head())


print("\n--- Plots ---")
plt.figure(figsize=(8, 6))

# sns.pointplot is perfect for interaction plots
sns.pointplot(
    data=df,
    x='Mode', 
    y='Error', 
    hue='Algorithm',       # Draws separate lines for each algorithm
    estimator=np.median,   # CRITICAL: Use median for non-parametric data
    errorbar=('ci', 95),   # Shows 95% confidence intervals
    capsize=0.1,           # Adds caps to the error bars
    dodge=0.1,             # Slightly offsets the points so they don't overlap
    markers=['o', 's'],    # Uses circles for AlgA, squares for AlgB
    linestyles=['-', '--'] # Solid line for AlgA, dashed for AlgB
)

plt.title('2-Way Interaction: Median Error by Mode and Algorithm')
plt.ylabel('Median Estimation Error')
plt.xlabel('Mode of Operation')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.show()

# sns.catplot allows us to create faceted grids easily
g = sns.catplot(
    data=df,
    kind='point',          # Tells catplot to draw point/line graphs
    x='Mode', 
    y='Error', 
    hue='Algorithm', 
    col='Scenario',        # CRITICAL: Creates a separate plot for each Scenario
    estimator=np.median,
    errorbar=('ci', 95),
    capsize=0.1,
    dodge=0.1,
    markers=['o', 's'],
    linestyles=['-', '--'],
    height=5,              # Height of each facet
    aspect=1.2             # Width-to-height ratio of each facet
)

# Add a main title to the entire figure
g.fig.suptitle('3-Way Interaction: Mode x Algorithm x Scenario', y=1.05)
g.set_axis_labels('Mode of Operation', 'Median Estimation Error')

plt.show()