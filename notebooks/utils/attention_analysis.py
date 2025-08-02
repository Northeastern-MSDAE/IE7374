# Additional Attention analyses

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

class AttentionAnalysis:
    
    def __init__(self, qa_pairs, attention_data):
        self.qa_pairs = qa_pairs
        self.attention_data = attention_data
        self.validate_data()
    
    def validate_data(self):
        """Validate input data and print diagnostics"""
        print("=== Data Validation ===")
        print(f"QA pairs shape: {self.qa_pairs.shape}")
        print(f"Attention data length: {len(self.attention_data)}")
        
        # Check for required columns
        required_cols = ['Domain', 't5_answer', 'gpt_answer']
        missing_cols = [col for col in required_cols if col not in self.qa_pairs.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
        
        # Check metric columns
        metrics = ['entropy', 'volatility', 'span', 'head_agreement', 'saas']
        for model in ['t5', 'gpt']:
            for metric in metrics:
                col = f'{model}_{metric}'
                if col in self.qa_pairs.columns:
                    non_null = self.qa_pairs[col].notna().sum()
                    print(f"{col}: {non_null}/{len(self.qa_pairs)} valid values")
                else:
                    print(f"Missing: {col}")
        
        # Check domains
        print(f"Domains: {self.qa_pairs['Domain'].value_counts().to_dict()}")
        print("========================\n")
    
    def attention_based_domain_classification(self):
        """Can we predict domain from attention patterns alone?"""
        # Flatten attention vectors
        X = np.array([self.flatten_attention(att) for att in self.attention_data])
        y = self.qa_pairs['Domain'].values
        
        # Cross-validation classification
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        
        print(f"Domain Classification from Attention: {scores.mean():.3f} +/- {scores.std():.3f}")
        
        # Feature importance analysis
        clf.fit(X, y)
        importance = clf.feature_importances_
        return scores, importance
    
    def attention_answer_quality_correlation(self):
        """Do attention metrics correlate with answer quality?"""
        # You'd need to implement answer quality scoring
        # For now, using answer length as proxy
        quality_proxy = pd.to_numeric(self.qa_pairs['t5_answer'].str.len(), errors='coerce')
        
        correlations = {}
        metrics = ['entropy', 'volatility', 'span', 'head_agreement', 'saas']
        
        for metric in metrics:
            metric_col = f't5_{metric}'
            if metric_col not in self.qa_pairs.columns:
                print(f"Warning: Column {metric_col} not found")
                continue
                
            # Convert to numeric and handle missing values
            metric_values = pd.to_numeric(self.qa_pairs[metric_col], errors='coerce')
            
            # Drop rows where either value is NaN
            valid_mask = ~(metric_values.isna() | quality_proxy.isna())
            
            if valid_mask.sum() < 3:  # Need at least 3 points for correlation
                print(f"Warning: Not enough valid data points for {metric}")
                correlations[metric] = {'correlation': np.nan, 'p_value': np.nan, 'n_valid': valid_mask.sum()}
                continue
            
            try:
                corr, p_val = stats.pearsonr(metric_values[valid_mask], quality_proxy[valid_mask])
                correlations[metric] = {
                    'correlation': corr, 
                    'p_value': p_val, 
                    'n_valid': valid_mask.sum()
                }
            except Exception as e:
                print(f"Error computing correlation for {metric}: {e}")
                correlations[metric] = {'correlation': np.nan, 'p_value': np.nan, 'n_valid': 0}
        
        return correlations
    
    def comparative_model_analysis(self):
        """Systematic comparison between T5 and GPT attention patterns"""
        metrics = ['entropy', 'volatility', 'span', 'head_agreement', 'saas']
        
        results = []
        for domain in ['Medical', 'Legal']:
            domain_data = self.qa_pairs[self.qa_pairs['Domain'] == domain]
            
            if len(domain_data) == 0:
                print(f"Warning: No data found for domain {domain}")
                continue
            
            for metric in metrics:
                t5_col = f't5_{metric}'
                gpt_col = f'gpt_{metric}'
                
                # Check if columns exist
                if t5_col not in domain_data.columns or gpt_col not in domain_data.columns:
                    print(f"Warning: Missing columns for {metric} in {domain}")
                    continue
                
                # Convert to numeric and handle missing values
                t5_vals = pd.to_numeric(domain_data[t5_col], errors='coerce').dropna()
                gpt_vals = pd.to_numeric(domain_data[gpt_col], errors='coerce').dropna()
                
                # Ensure we have matching pairs (same indices)
                common_idx = t5_vals.index.intersection(gpt_vals.index)
                if len(common_idx) < 3:
                    print(f"Warning: Not enough paired data for {metric} in {domain}")
                    continue
                
                t5_paired = t5_vals.loc[common_idx]
                gpt_paired = gpt_vals.loc[common_idx]
                
                try:
                    # Paired t-test (same questions answered by both models)
                    stat, p_val = stats.ttest_rel(t5_paired, gpt_paired)
                    pooled_std = np.sqrt((t5_paired.var() + gpt_paired.var()) / 2)
                    effect_size = (t5_paired.mean() - gpt_paired.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    results.append({
                        'domain': domain,
                        'metric': metric,
                        'mean_diff': t5_paired.mean() - gpt_paired.mean(),
                        'p_value': p_val,
                        'effect_size': effect_size,
                        'n_pairs': len(common_idx)
                    })
                except Exception as e:
                    print(f"Error in analysis for {metric} in {domain}: {e}")
        
        return pd.DataFrame(results)
    
    def attention_pattern_clustering(self):
        """Unsupervised clustering of attention patterns"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        
        try:
            X = np.array([self.flatten_attention(att) for att in self.attention_data])
            
            # Check for valid data
            if X.shape[0] == 0 or np.any(np.isnan(X)) or np.any(np.isinf(X)):
                print("Warning: Invalid attention data for clustering")
                return None, 0.0
            
            # K-means clustering
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            predicted_clusters = kmeans.fit_predict(X)
            
            # Compare with true domain labels
            true_labels = (self.qa_pairs['Domain'] == 'Medical').astype(int)
            
            # Ensure same length
            min_len = min(len(predicted_clusters), len(true_labels))
            predicted_clusters = predicted_clusters[:min_len]
            true_labels = true_labels[:min_len]
            
            ari_score = adjusted_rand_score(true_labels, predicted_clusters)
            
            print(f"Attention Pattern Clustering ARI: {ari_score:.3f}")
            return predicted_clusters, ari_score
            
        except Exception as e:
            print(f"Error in clustering analysis: {e}")
            return None, 0.0
    
    def statistical_power_analysis(self):
        """Assess whether sample size is adequate"""
        from scipy.stats import power
        
        # Example: power analysis for domain differences in entropy
        medical_entropy = self.qa_pairs[self.qa_pairs['Domain'] == 'Medical']['t5_entropy']
        legal_entropy = self.qa_pairs[self.qa_pairs['Domain'] == 'Legal']['t5_entropy']
        
        effect_size = abs(medical_entropy.mean() - legal_entropy.mean()) / np.sqrt(
            (medical_entropy.var() + legal_entropy.var()) / 2
        )
        
        # Calculate achieved power
        n = min(len(medical_entropy), len(legal_entropy))
        # using manual calculation
        
        return effect_size, n
    
    def attention_head_specialization(self):
        """Analyze which attention heads are most domain-specific"""
        # This requires per-head analysis rather than averaged attention
        head_specificity = {}
        
        for i, attention_tensor in enumerate(self.attention_data):
            # attention_tensor is list of [H, seq_out, seq_in] tensors per layer
            domain = self.qa_pairs.iloc[i]['Domain']
            
            for layer_idx, layer_att in enumerate(attention_tensor):
                for head_idx in range(layer_att.shape[0]):
                    head_key = f"L{layer_idx}_H{head_idx}"
                    if head_key not in head_specificity:
                        head_specificity[head_key] = {'Medical': [], 'Legal': []}
                    
                    head_pattern = layer_att[head_idx].flatten().cpu().numpy()
                    head_specificity[head_key][domain].append(head_pattern.mean())
        
        return head_specificity
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        report = {
            'classification_accuracy': self.attention_based_domain_classification(),
            'quality_correlations': self.attention_answer_quality_correlation(),
            'model_comparisons': self.comparative_model_analysis(),
            'clustering_results': self.attention_pattern_clustering(),
            'power_analysis': self.statistical_power_analysis(),
            'head_specialization': self.attention_head_specialization()
        }
        
        return report
    
    def flatten_attention(self, attention_list):
        """Helper to flatten attention tensors to vectors"""
        # Concatenate and flatten all attention layers
        flattened = []
        for layer_att in attention_list:
            flattened.append(layer_att.mean(dim=0).flatten().cpu().numpy())
        return np.concatenate(flattened)