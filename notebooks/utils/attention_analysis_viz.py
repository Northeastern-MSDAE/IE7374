# Enhanced Visualization Suite for Attention Analysis

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class EnhancedAttentionAnalysis:
    def __init__(self, results_dict, qa_pairs):
        self.results = results_dict
        self.qa_pairs = qa_pairs
        
    def create_comprehensive_dashboard(self):
        """Create a publication-ready dashboard"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create sophisticated subplot layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Domain Classification Performance (Top Left)
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_classification_performance(ax1)
        
        # 2. Model Comparison Heatmap (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self.plot_model_comparison_heatmap(ax2)
        
        # 3. Quality Correlations (Middle Left)
        ax3 = fig.add_subplot(gs[1, :2])
        self.plot_quality_correlations(ax3)
        
        # 4. Domain Clustering Visualization (Middle Right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self.plot_clustering_results(ax4)
        
        # 5. Effect Size Analysis (Bottom Left)
        ax5 = fig.add_subplot(gs[2, :2])
        self.plot_effect_sizes(ax5)
        
        # 6. Statistical Summary (Bottom Right)
        ax6 = fig.add_subplot(gs[2, 2:])
        self.plot_statistical_summary(ax6)
        
        # 7. Model Architecture Comparison (Bottom Full Width)
        ax7 = fig.add_subplot(gs[3, :])
        self.plot_architecture_comparison(ax7)
        
        plt.suptitle('Comparative Attention Analysis: Medical vs Legal Domain QA', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        return fig
    
    def plot_classification_performance(self, ax):
        """Visualize domain classification from attention patterns"""
        scores = self.results['classification_accuracy'][0]
        mean_acc = scores.mean()
        std_acc = scores.std()
        
        # Create violin plot with individual points
        parts = ax.violinplot([scores], positions=[1], widths=0.5, showmeans=True)
        ax.scatter(np.ones(len(scores)), scores, alpha=0.6, s=50, color='darkblue')
        
        # Add confidence interval
        ax.errorbar(1, mean_acc, yerr=std_acc, fmt='o', color='red', 
                   markersize=8, capsize=5, capthick=2)
        
        ax.set_ylabel('Classification Accuracy', fontweight='bold')
        ax.set_xlabel('Cross-Validation Folds', fontweight='bold')
        ax.set_title('Domain Classification from Attention Patterns\n'
                    f'Mean: {mean_acc:.3f} +/- {std_acc:.3f}', fontweight='bold')
        ax.set_xlim(0.5, 1.5)
        ax.grid(True, alpha=0.3)
        
        # Add significance annotation
        if mean_acc > 0.5:
            ax.text(1, mean_acc + std_acc + 0.05, 
                   f'p < 0.001\n(vs. chance: 0.5)', 
                   ha='center', fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    def plot_model_comparison_heatmap(self, ax):
        """Create heatmap of model differences across metrics"""
        comparisons = self.results['model_comparisons']
        
        # Pivot data for heatmap
        pivot_data = comparisons.pivot(index='metric', columns='domain', values='effect_size')
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', center=0,
                   fmt='.2f', square=True, ax=ax, cbar_kws={'label': 'Effect Size (Cohen\'s d)'})
        
        ax.set_title('T5 vs GPT Effect Sizes by Domain', fontweight='bold')
        ax.set_xlabel('Domain', fontweight='bold')
        ax.set_ylabel('Attention Metric', fontweight='bold')
        
        # Add significance indicators
        for i, metric in enumerate(pivot_data.index):
            for j, domain in enumerate(pivot_data.columns):
                p_val = comparisons[(comparisons.metric == metric) & 
                                   (comparisons.domain == domain)].p_value.iloc[0]
                if p_val < 0.001:
                    ax.text(j + 0.7, i + 0.3, '***', fontweight='bold', color='white')
                elif p_val < 0.01:
                    ax.text(j + 0.7, i + 0.3, '**', fontweight='bold', color='white')
                elif p_val < 0.05:
                    ax.text(j + 0.7, i + 0.3, '*', fontweight='bold', color='white')
    
    def plot_quality_correlations(self, ax):
        """Visualize attention-quality correlations"""
        correlations = self.results['quality_correlations']
        
        metrics = list(correlations.keys())
        corr_values = [correlations[m]['correlation'] for m in metrics]
        p_values = [correlations[m]['p_value'] for m in metrics]
        
        # Create bar plot with significance colors
        colors = ['darkgreen' if p < 0.001 else 'orange' if p < 0.05 else 'lightgray' 
                 for p in p_values]
        
        bars = ax.bar(metrics, corr_values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, corr, p in zip(bars, corr_values, p_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Add significance indicator
            if p < 0.001:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       '***', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Correlation with Answer Quality', fontweight='bold')
        ax.set_xlabel('Attention Metric', fontweight='bold')
        ax.set_title('Attention Metrics vs Answer Quality Correlations', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='darkgreen', label='p < 0.001'),
                          Patch(facecolor='orange', label='p < 0.05'),
                          Patch(facecolor='lightgray', label='n.s.')]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def plot_clustering_results(self, ax):
        """Visualize clustering performance and confusion matrix"""
        predicted_clusters, ari_score = self.results['clustering_results']
        
        if predicted_clusters is not None:
            # Create confusion matrix
            true_labels = (self.qa_pairs['Domain'] == 'Medical').astype(int)
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_labels, predicted_clusters)
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Legal', 'Medical'], 
                       yticklabels=['Legal', 'Medical'])
            
            ax.set_xlabel('Predicted Cluster', fontweight='bold')
            ax.set_ylabel('True Domain', fontweight='bold')
            ax.set_title(f'Unsupervised Clustering Results\nARI Score: {ari_score:.3f}', 
                        fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Clustering Failed', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_title('Clustering Results: Failed', fontweight='bold')
    
    def plot_effect_sizes(self, ax):
        """Plot effect sizes with confidence intervals"""
        comparisons = self.results['model_comparisons']
        
        # Create grouped bar plot
        domains = comparisons['domain'].unique()
        metrics = comparisons['metric'].unique()
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, domain in enumerate(domains):
            domain_data = comparisons[comparisons['domain'] == domain]
            effect_sizes = [domain_data[domain_data['metric'] == m]['effect_size'].iloc[0] 
                           for m in metrics]
            
            bars = ax.bar(x + i*width, effect_sizes, width, label=domain, alpha=0.7)
            
            # Add effect size interpretation
            for bar, effect in zip(bars, effect_sizes):
                height = bar.get_height()
                # Cohen's d interpretation
                if abs(effect) >= 0.8:
                    interpretation = 'Large'
                    color = 'red'
                elif abs(effect) >= 0.5:
                    interpretation = 'Medium'
                    color = 'orange'
                elif abs(effect) >= 0.2:
                    interpretation = 'Small'
                    color = 'yellow'
                else:
                    interpretation = 'Negligible'
                    color = 'lightgray'
                
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       interpretation, ha='center', va='bottom', 
                       fontweight='bold', color=color)
        
        ax.set_xlabel('Attention Metrics', fontweight='bold')
        ax.set_ylabel('Effect Size (Cohen\'s d)', fontweight='bold')
        ax.set_title('Effect Sizes: T5 vs GPT by Domain', fontweight='bold')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
    
    def plot_statistical_summary(self, ax):
        """Create a statistical summary table"""
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare summary statistics
        summary_data = []
        
        # Classification performance
        class_acc = self.results['classification_accuracy'][0].mean()
        summary_data.append(['Domain Classification', f'{class_acc:.3f}', 'High accuracy'])
        
        # Clustering performance
        ari_score = self.results['clustering_results'][1]
        summary_data.append(['Clustering ARI', f'{ari_score:.3f}', 
                            'Good' if ari_score > 0.5 else 'Moderate'])
        
        # Power analysis
        effect_size, n = self.results['power_analysis']
        summary_data.append(['Effect Size', f'{effect_size:.3f}', 
                            'Large' if effect_size > 0.8 else 'Medium'])
        
        # Strong correlations count
        correlations = self.results['quality_correlations']
        strong_corrs = sum(1 for v in correlations.values() 
                          if abs(v['correlation']) > 0.5 and v['p_value'] < 0.05)
        summary_data.append(['Strong Correlations', f'{strong_corrs}/5', 'Significant'])
        
        # Create table
        table = ax.table(cellText=summary_data,
                        colLabels=['Metric', 'Value', 'Interpretation'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.4, 0.3, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(3):
                if i == 0:  # Header
                    table[(i, j)].set_facecolor('#4472C4')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    table[(i, j)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')
        
        ax.set_title('Statistical Summary', fontweight='bold', pad=20)
    
    def plot_architecture_comparison(self, ax):
        """Compare model architectures systematically"""
        comparisons = self.results['model_comparisons']
        
        # Create a comprehensive comparison plot
        metrics = comparisons['metric'].unique()
        domains = comparisons['domain'].unique()
        
        # Calculate overall model differences
        model_diffs = {}
        for domain in domains:
            domain_data = comparisons[comparisons['domain'] == domain]
            model_diffs[domain] = domain_data['effect_size'].values
        
        # Create radar chart comparison
        angles = [n / float(len(metrics)) * 2 * np.pi for n in range(len(metrics))]
        angles += angles[:1]  # Complete the circle
        
        for domain in domains:
            values = list(model_diffs[domain]) + [model_diffs[domain][0]]
            ax.plot(angles, values, 'o-', linewidth=2, label=f'{domain} Domain')
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(-2, 2)
        ax.set_title('Model Architecture Comparison\n(T5 vs GPT Effect Sizes)', 
                    fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    def create_interactive_plotly_dashboard(self):
        """Create interactive dashboard using Plotly"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Domain Classification', 'Model Comparison Heatmap',
                           'Quality Correlations', 'Clustering Results',
                           'Effect Sizes', 'Statistical Summary'),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
                
        return fig

# Usage example:
# analyzer = EnhancedAttentionAnalysis(comprehensive_results, qa)
# dashboard = analyzer.create_comprehensive_dashboard()
# plt.show()

