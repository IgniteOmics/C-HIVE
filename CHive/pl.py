import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def t_stat_heatmap(t_stat_matrix, save_path=None):
    ax = sns.heatmap(
        t_stat_matrix.astype(float),
        cmap='coolwarm',
        center=0,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return ax

def topic_genes(weight_matrix: pd.DataFrame,
    topic_name: str,
    top_n: int = 20,
    save_path: str = None,
    title: str = None
):
    if topic_name not in weight_matrix.index:
        raise ValueError(f"Topic not found: {topic_name}")

    topic_weights = weight_matrix.loc[topic_name]
    top_genes = topic_weights.sort_values(ascending=False).head(top_n)

    ax = plt.gca()
    ax.scatter(top_genes.index, top_genes.values, color='blue')
    ax.set_ylabel("Weight")
    ax.set_xticklabels(top_genes.index, rotation=45)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return ax
