# Additional specialized visualizations

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_attention_head_analysis(attention_data, qa_pairs):
    """Proper attention head specialization analysis"""
    
    def compute_head_variance(attention_tensors):
        """Compute variance across domains for each head"""
        head_variances = {}
        
        for i, attn_tensor in enumerate(attention_tensors):
            domain = qa_pairs.iloc[i]['Domain']
            
            for layer_idx, layer_att in enumerate(attn_tensor):
                for head_idx in range(layer_att.shape[0]):
                    head_key = f"L{layer_idx}_H{head_idx}"
                    
                    if head_key not in head_variances:
                        head_variances[head_key] = {'Medical': [], 'Legal': []}
                    
                    # Compute attention entropy for this head
                    head_pattern = layer_att[head_idx].cpu().numpy()
                    entropy = -np.sum(head_pattern * np.log(head_pattern + 1e-12))
                    head_variances[head_key][domain].append(entropy)
        
        return head_variances
    
    # Create heatmap of head specialization
    head_data = compute_head_variance(attention_data)
    
    # Convert to matrix for visualization
    layers = 12  # T5-small has 12 layers
    heads = 12   # 12 heads per layer
    
    specialization_matrix = np.zeros((layers, heads))
    
    for layer in range(layers):
        for head in range(heads):
            head_key = f"L{layer}_H{head}"
            if head_key in head_data:
                med_vals = head_data[head_key]['Medical']
                leg_vals = head_data[head_key]['Legal']
                
                if len(med_vals) > 0 and len(leg_vals) > 0:
                    # Use t-test to measure specialization
                    from scipy.stats import ttest_ind
                    _, p_val = ttest_ind(med_vals, leg_vals)
                    specialization_matrix[layer, head] = -np.log10(p_val + 1e-10)
    
    # Plot heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(specialization_matrix, 
                xticklabels=[f'H{i}' for i in range(heads)],
                yticklabels=[f'L{i}' for i in range(layers)],
                cmap='viridis', cbar_kws={'label': '-log10(p-value)'})
    plt.title('Attention Head Specialization by Domain\n(Higher values = more domain-specific)', 
              fontweight='bold')
    plt.xlabel('Attention Head', fontweight='bold')
    plt.ylabel('Layer', fontweight='bold')
    plt.show()


# Better head specialization analysis
def analyze_head_specialization_proper(attention_data, qa_pairs):
    head_domain_differences = {}
    
    for i, attention_tensors in enumerate(attention_data):
        domain = qa_pairs.iloc[i]['Domain']
        
        for layer_idx, layer_att in enumerate(attention_tensors):
            for head_idx in range(layer_att.shape[0]):
                head_key = f"L{layer_idx}_H{head_idx}"
                
                if head_key not in head_domain_differences:
                    head_domain_differences[head_key] = {'Medical': [], 'Legal': []}
                
                # Use attention entropy or max attention as the measure
                head_pattern = layer_att[head_idx].cpu().numpy()
                attention_entropy = -np.sum(head_pattern * np.log(head_pattern + 1e-12), axis=-1).mean()
                head_domain_differences[head_key][domain].append(attention_entropy)
    
    return head_domain_differences