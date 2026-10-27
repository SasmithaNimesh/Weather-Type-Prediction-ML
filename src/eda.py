"""Exploratory Data Analysis (EDA) Module for Weather Type Prediction.

Analyzes input variables, class distributions, pairwise correlations,
and categorical relationships, saving publication-quality plots to reports/figures/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

WEATHER_PALETTE = {
    'Sunny': '#FDB813',
    'Cloudy': '#95A5A6',
    'Rainy': '#3498DB',
    'Snowy': '#9B59B6'
}


def load_dataset(file_path: str = "data/weather_classification_data.csv") -> pd.DataFrame:
    """Load dataset from CSV file with error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset successfully loaded. Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def generate_summary_statistics(df: pd.DataFrame, output_dir: str = "reports") -> None:
    """Print and save descriptive statistics."""
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "dataset_summary.txt")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("WEATHER TYPE DATASET - STATISTICAL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Observations: {df.shape[0]:,}\n")
        f.write(f"Total Features:     {df.shape[1]}\n\n")
        
        f.write("--- Missing Values ---\n")
        f.write(df.isnull().sum().to_string() + "\n\n")
        
        f.write("--- Target Distribution ('Weather Type') ---\n")
        target_counts = df['Weather Type'].value_counts()
        target_pct = df['Weather Type'].value_counts(normalize=True) * 100
        target_df = pd.DataFrame({'Count': target_counts, 'Percentage (%)': target_pct})
        f.write(target_df.to_string() + "\n\n")
        
        f.write("--- Numerical Features Summary ---\n")
        f.write(df.describe().round(2).to_string() + "\n\n")
        
        f.write("--- Categorical Features Unique Values ---\n")
        cat_cols = df.select_dtypes(include=['object', 'string']).columns.drop('Weather Type', errors='ignore')
        for col in cat_cols:
            f.write(f"\nFeature [{col}]:\n")
            f.write(df[col].value_counts().to_string() + "\n")
            
    print(f"Summary statistics saved to: {summary_path}")


def plot_target_distribution(df: pd.DataFrame, figures_dir: str) -> None:
    """Plot target class distribution (Bar & Donut)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df['Weather Type'].value_counts()
    colors = [WEATHER_PALETTE.get(w, '#4C72B0') for w in counts.index]
    
    # Bar Chart
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title("Weather Type Class Counts", fontsize=14, fontweight='bold', pad=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_xlabel("Weather Category", fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + 50, f"{int(yval):,}", ha='center', va='bottom', fontweight='bold')
    
    # Donut Chart
    wedges, texts, autotexts = axes[1].pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.75,
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops=dict(width=0.45, edgecolor='black')
    )
    axes[1].set_title("Target Proportion (Balanced Split)", fontsize=14, fontweight='bold', pad=12)
    
    plt.tight_layout()
    save_path = os.path.join(figures_dir, "01_class_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_numerical_distributions(df: pd.DataFrame, figures_dir: str) -> None:
    """Plot distribution of each numerical feature."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='#2980B9', edgecolor='black', alpha=0.6)
        axes[i].set_title(f"Distribution: {col}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel("Density / Frequency", fontsize=10)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("Feature Distributions (Numerical Variables)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    save_path = os.path.join(figures_dir, "02_numerical_distributions.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_features_by_weather_type(df: pd.DataFrame, figures_dir: str) -> None:
    """Boxplots comparing numerical variables across weather classes."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        sns.boxplot(
            data=df,
            x='Weather Type',
            y=col,
            ax=axes[i],
            palette=WEATHER_PALETTE,
            boxprops=dict(alpha=0.85),
            fliersize=2
        )
        axes[i].set_title(f"{col} by Weather Type", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("")
        axes[i].set_ylabel(col, fontsize=10)
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("Numerical Variables Partitioned by Weather Category", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    save_path = os.path.join(figures_dir, "03_boxplots_by_weather.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, figures_dir: str) -> None:
    """Plot correlation matrix between numerical features."""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr(method='pearson')
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=.6,
        annot=True,
        fmt='.2f',
        cbar_kws={"shrink": .8}
    )
    plt.title("Pearson Correlation Heatmap (Numerical Features)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    save_path = os.path.join(figures_dir, "04_correlation_heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_categorical_breakdowns(df: pd.DataFrame, figures_dir: str) -> None:
    """Grouped bar plots showing categorical associations with weather types."""
    cat_cols = ['Cloud Cover', 'Season', 'Location']
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(18, 5))
    if len(cat_cols) == 1:
        axes = [axes]
        
    for i, col in enumerate(cat_cols):
        cross_tab = pd.crosstab(df[col], df['Weather Type'], normalize='index') * 100
        cross_tab.plot(
            kind='bar',
            stacked=True,
            ax=axes[i],
            color=[WEATHER_PALETTE.get(w, '#4C72B0') for w in cross_tab.columns],
            edgecolor='black',
            alpha=0.85
        )
        axes[i].set_title(f"Weather Distribution across {col}", fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Proportion (%)")
        axes[i].set_xlabel(col)
        axes[i].legend(title="Weather Type", bbox_to_anchor=(1.02, 1), loc='upper left')
        axes[i].tick_params(axis='x', rotation=30)
        
    plt.suptitle("Categorical Predictors vs. Weather Categories", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(figures_dir, "05_categorical_breakdown.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def run_eda(data_path: str = "data/weather_classification_data.csv", output_dir: str = "reports") -> pd.DataFrame:
    """Execute complete EDA pipeline and generate all reports."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("STARTING EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 50)
    
    df = load_dataset(data_path)
    generate_summary_statistics(df, output_dir)
    plot_target_distribution(df, figures_dir)
    plot_numerical_distributions(df, figures_dir)
    plot_features_by_weather_type(df, figures_dir)
    plot_correlation_heatmap(df, figures_dir)
    plot_categorical_breakdowns(df, figures_dir)
    
    print("=" * 50)
    print("EDA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50 + "\n")
    return df


if __name__ == "__main__":
    run_eda()
