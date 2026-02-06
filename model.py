import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, r2_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import copy
import time
import warnings
warnings.filterwarnings("ignore")

EXPRESSION_FILE = "exp.csv"
PPI_FILE = "ppi.csv"
PATHWAY_FILE = "pat.csv"
OUTPUT_DIR = "results"
TOP_N_TARGETS = 20
RANDOM_SEED = 100
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def load_gene_expression(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df.columns) == 1:
            df.columns = ['Gene']
            df['Log2FC'] = 1.0
            print("Detected single-column gene list format. Assigning default Log2FC value of 1.0 to all genes.")
            return df
        if 'Target' in df.columns and 'Log2FC' in df.columns:
            df = df[['Target', 'Log2FC']]
            df.columns = ['Gene', 'Log2FC']
        elif len(df.columns) >= 2:
            df.columns = ['Gene', 'Log2FC'] + list(df.columns[2:])
            df = df[['Gene', 'Log2FC']]
        else:
            raise ValueError("Expression CSV format not recognized. Please provide either a single column with gene symbols or at least two columns with Target and Log2FC values.")
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Error loading gene expression data: {e}")
        return pd.DataFrame(columns=['Gene', 'Log2FC'])

def load_ppi_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'node1' not in df.columns or 'node2' not in df.columns or 'combined_score' not in df.columns:
            if df.shape[1] >= 11:
                df.columns = ['node1', 'node2', 'neighborhood_on_chromosome', 'gene_fusion', 'phylogenetic_cooccurrence', 'homology', 'coexpression', 'experimentally_determined_interaction', 'database_annotated', 'automated_textmining', 'combined_score']
            else:
                raise ValueError("PPI CSV should have at least node1, node2, and combined_score columns")
        df = df[['node1', 'node2', 'combined_score']]
        df.columns = ['Gene1', 'Gene2', 'Combined_Score']
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Error loading PPI data: {e}")
        return pd.DataFrame(columns=['Gene1', 'Gene2', 'Combined_Score'])

def load_pathway_genes(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df.columns) == 1:
            genes = df[df.columns[0]].tolist()
        else:
            genes = []
            for col in df.columns:
                genes.extend(df[col].dropna().tolist())
        genes = [gene for gene in genes if isinstance(gene, str)]
        return list(set(genes))
    except Exception as e:
        print(f"Warning: Pathway data not available or error loading: {e}")
        return []

def create_features(gene_exp_df, ppi_df, pathway_genes=None):
    G = nx.Graph()
    for _, row in ppi_df.iterrows():
        G.add_edge(row['Gene1'], row['Gene2'], weight=row['Combined_Score'])
    all_genes = set(gene_exp_df['Gene'].unique())
    all_genes.update(G.nodes())
    for gene in all_genes:
        if gene not in G:
            G.add_node(gene)
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    avg_combined_scores = {}
    for gene in G.nodes():
        neighbors = list(G.neighbors(gene))
        if neighbors:
            total_score = sum(G[gene][neighbor]['weight'] for neighbor in neighbors)
            avg_combined_scores[gene] = total_score / len(neighbors)
        else:
            avg_combined_scores[gene] = 0.0
    features = []
    has_pathway_data = pathway_genes is not None and len(pathway_genes) > 0
    for gene in all_genes:
        gene_exp_entries = gene_exp_df[gene_exp_df['Gene'] == gene]
        log2fc = gene_exp_entries['Log2FC'].mean() if not gene_exp_entries.empty else 0
        degree = degree_centrality.get(gene, 0)
        betweenness = betweenness_centrality.get(gene, 0)
        closeness = closeness_centrality.get(gene, 0)
        avg_combined_score = avg_combined_scores.get(gene, 0)
        in_pathway = 1 if has_pathway_data and gene in pathway_genes else 0
        feature_dict = {
            'Gene': gene,
            'Log2FC': log2fc,
            'Degree': degree,
            'Betweenness': betweenness,
            'Closeness': closeness,
            'AvgCombinedScore': avg_combined_score
        }
        if has_pathway_data:
            feature_dict['InPathway'] = in_pathway
        features.append(feature_dict)
    return pd.DataFrame(features), G

class TargetIdentificationModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super(TargetIdentificationModel, self).__init__()
        self.conv1 = GCNConv(feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)

def train_single_run(model, data, optimizer, loss_fn, num_epochs=200, verbose=False):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_state = None
    epoch_metrics = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': [],
        'train_r2': [], 'val_r2': []
    }
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            train_preds = out[data.train_mask]
            val_preds = out[data.val_mask]
            train_loss = loss_fn(train_preds, data.y[data.train_mask]).item()
            val_loss = loss_fn(val_preds, data.y[data.val_mask]).item()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            try:
                train_r2 = r2_score(data.y[data.train_mask].cpu(), train_preds.cpu())
            except:
                train_r2 = float('nan')
            try:
                val_r2 = r2_score(data.y[data.val_mask].cpu(), val_preds.cpu())
            except:
                val_r2 = float('nan')
            train_binary_preds = (train_preds >= 0.5).float()
            val_binary_preds = (val_preds >= 0.5).float()
            train_binary_targets = (data.y[data.train_mask] >= 0.5).float()
            val_binary_targets = (data.y[data.val_mask] >= 0.5).float()
            train_acc = (train_binary_preds == train_binary_targets).float().mean().item()
            val_acc = (val_binary_preds == val_binary_targets).float().mean().item()
            train_prec = precision_score(train_binary_targets.cpu(), train_binary_preds.cpu(), zero_division=0)
            val_prec = precision_score(val_binary_targets.cpu(), val_binary_preds.cpu(), zero_division=0)
            train_rec = recall_score(train_binary_targets.cpu(), train_binary_preds.cpu(), zero_division=0)
            val_rec = recall_score(val_binary_targets.cpu(), val_binary_preds.cpu(), zero_division=0)
            train_f1 = f1_score(train_binary_targets.cpu(), train_binary_preds.cpu(), zero_division=0)
            val_f1 = f1_score(val_binary_targets.cpu(), val_binary_preds.cpu(), zero_division=0)
            epoch_metrics['train_loss'].append(train_loss)
            epoch_metrics['val_loss'].append(val_loss)
            epoch_metrics['train_acc'].append(train_acc)
            epoch_metrics['val_acc'].append(val_acc)
            epoch_metrics['train_precision'].append(train_prec)
            epoch_metrics['val_precision'].append(val_prec)
            epoch_metrics['train_recall'].append(train_rec)
            epoch_metrics['val_recall'].append(val_rec)
            epoch_metrics['train_f1'].append(train_f1)
            epoch_metrics['val_f1'].append(val_f1)
            epoch_metrics['train_r2'].append(train_r2)
            epoch_metrics['val_r2'].append(val_r2)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, epoch_metrics, best_val_loss

def build_data_object_from_features(features_df, G, train_idx=None, val_idx=None, test_idx=None):
    feature_columns = ['Log2FC', 'Degree', 'Betweenness', 'Closeness', 'AvgCombinedScore']
    if 'InPathway' in features_df.columns:
        feature_columns.append('InPathway')
    x = torch.tensor(features_df[feature_columns].values, dtype=torch.float)
    y = torch.tensor(features_df['ImportanceScore'].values, dtype=torch.float).view(-1, 1)
    edge_index = []
    edge_attr = []
    node_mapping = {gene: i for i, gene in enumerate(features_df['Gene'])}
    for u, v, data in G.edges(data=True):
        if u in node_mapping and v in node_mapping:
            edge_index.append([node_mapping[u], node_mapping[v]])
            edge_index.append([node_mapping[v], node_mapping[u]])
            edge_attr.append(data['weight'])
            edge_attr.append(data['weight'])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.zeros(0, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    num_nodes = data.x.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if train_idx is not None:
        train_mask[train_idx] = True
    if val_idx is not None:
        val_mask[val_idx] = True
    if test_idx is not None:
        test_mask[test_idx] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data

def evaluate_model_numeric(model, data):
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index, data.edge_attr)
        predictions = predictions.cpu().numpy().reshape(-1,)
        labels = data.y.cpu().numpy().reshape(-1,)
        train_mask = data.train_mask.cpu().numpy()
        val_mask = data.val_mask.cpu().numpy()
        test_mask = data.test_mask.cpu().numpy()
        metrics = {}
        train_pred = predictions[train_mask]
        train_true = labels[train_mask]
        if train_pred.size > 0:
            metrics['train_mse'] = mean_squared_error(train_true, train_pred)
            metrics['train_r2'] = r2_score(train_true, train_pred) if len(np.unique(train_true)) > 1 else float('nan')
            metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
            train_binary_pred = (train_pred >= 0.5).astype(int)
            train_binary_true = (train_true >= 0.5).astype(int)
            metrics['train_accuracy'] = accuracy_score(train_binary_true, train_binary_pred)
            metrics['train_precision'] = precision_score(train_binary_true, train_binary_pred, zero_division=0)
            metrics['train_recall'] = recall_score(train_binary_true, train_binary_pred, zero_division=0)
            metrics['train_f1'] = f1_score(train_binary_true, train_binary_pred, zero_division=0)
            if np.unique(train_binary_true).size > 1:
                metrics['train_auc_roc'] = roc_auc_score(train_binary_true, train_pred)
                metrics['train_auc_pr'] = average_precision_score(train_binary_true, train_pred)
            else:
                metrics['train_auc_roc'] = float('nan')
                metrics['train_auc_pr'] = float('nan')
            metrics['train_confusion_matrix'] = confusion_matrix(train_binary_true, train_binary_pred)
        else:
            metrics.update({k: float('nan') for k in ['train_mse','train_r2','train_rmse','train_accuracy','train_precision','train_recall','train_f1','train_auc_roc','train_auc_pr']})
            metrics['train_confusion_matrix'] = np.array([[0,0],[0,0]])
        val_pred = predictions[val_mask]
        val_true = labels[val_mask]
        if val_pred.size > 0:
            metrics['val_mse'] = mean_squared_error(val_true, val_pred)
            metrics['val_r2'] = r2_score(val_true, val_pred) if len(np.unique(val_true)) > 1 else float('nan')
            metrics['val_rmse'] = np.sqrt(metrics['val_mse'])
            val_binary_pred = (val_pred >= 0.5).astype(int)
            val_binary_true = (val_true >= 0.5).astype(int)
            metrics['val_accuracy'] = accuracy_score(val_binary_true, val_binary_pred)
            metrics['val_precision'] = precision_score(val_binary_true, val_binary_pred, zero_division=0)
            metrics['val_recall'] = recall_score(val_binary_true, val_binary_pred, zero_division=0)
            metrics['val_f1'] = f1_score(val_binary_true, val_binary_pred, zero_division=0)
            if np.unique(val_binary_true).size > 1:
                metrics['val_auc_roc'] = roc_auc_score(val_binary_true, val_pred)
                metrics['val_auc_pr'] = average_precision_score(val_binary_true, val_pred)
            else:
                metrics['val_auc_roc'] = float('nan')
                metrics['val_auc_pr'] = float('nan')
            metrics['val_confusion_matrix'] = confusion_matrix(val_binary_true, val_binary_pred)
        else:
            metrics.update({k: float('nan') for k in ['val_mse','val_r2','val_rmse','val_accuracy','val_precision','val_recall','val_f1','val_auc_roc','val_auc_pr']})
            metrics['val_confusion_matrix'] = np.array([[0,0],[0,0]])
        test_pred = predictions[test_mask]
        test_true = labels[test_mask]
        if test_pred.size > 0:
            metrics['test_mse'] = mean_squared_error(test_true, test_pred)
            metrics['test_r2'] = r2_score(test_true, test_pred) if len(np.unique(test_true)) > 1 else float('nan')
            metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
            test_binary_pred = (test_pred >= 0.5).astype(int)
            test_binary_true = (test_true >= 0.5).astype(int)
            metrics['test_accuracy'] = accuracy_score(test_binary_true, test_binary_pred)
            metrics['test_precision'] = precision_score(test_binary_true, test_binary_pred, zero_division=0)
            metrics['test_recall'] = recall_score(test_binary_true, test_binary_pred, zero_division=0)
            metrics['test_f1'] = f1_score(test_binary_true, test_binary_pred, zero_division=0)
            if np.unique(test_binary_true).size > 1:
                metrics['test_auc_roc'] = roc_auc_score(test_binary_true, test_pred)
                metrics['test_auc_pr'] = average_precision_score(test_binary_true, test_pred)
            else:
                metrics['test_auc_roc'] = float('nan')
                metrics['test_auc_pr'] = float('nan')
            metrics['test_confusion_matrix'] = confusion_matrix(test_binary_true, test_binary_pred)
        else:
            metrics.update({k: float('nan') for k in ['test_mse','test_r2','test_rmse','test_accuracy','test_precision','test_recall','test_f1','test_auc_roc','test_auc_pr']})
            metrics['test_confusion_matrix'] = np.array([[0,0],[0,0]])
        return metrics

def rank_targets(predictions_df, top_n=10):
    ranked_targets = predictions_df.sort_values('PredictedScore', ascending=False)
    return ranked_targets.head(top_n)

def visualize_network(G, top_targets_df, output_file=None):
    plt.figure(figsize=(12, 10), dpi=300)
    top_genes = set(top_targets_df['Gene'])
    nodes_to_include = set()
    for gene in top_genes:
        nodes_to_include.add(gene)
        if gene in G:
            nodes_to_include.update(G.neighbors(gene))
    subgraph = G.subgraph(nodes_to_include)
    node_colors = []
    for node in subgraph.nodes():
        if node in top_genes:
            node_colors.append('red')
        else:
            node_colors.append('blue')
    node_sizes = []
    for node in subgraph.nodes():
        if node in top_genes:
            node_sizes.append(300)
        else:
            node_sizes.append(100)
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
    edge_widths = []
    for u, v, data in subgraph.edges(data=True):
        edge_widths.append(data['weight'] * 3)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.3, width=edge_widths)
    nx.draw_networkx_labels(subgraph, pos, font_size=8)
    plt.title("Protein-Protein Interaction Network - Top Therapeutic Targets", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.axis('off')
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Network visualization saved to {output_file}")
    else:
        plt.show()

def save_results(top_targets, output_file):
    top_targets.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def plot_training_validation_curves(train_losses, val_losses, output_file=None):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontweight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Loss', fontweight="bold", fontsize=14, labelpad=20)
    plt.title('Training and Validation Loss', fontweight="bold", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Training and validation loss curves saved to {output_file}")
    else:
        plt.savefig("Training_and_Validation_Loss.png", dpi=300)
    plt.close()

def plot_accuracy_curves(train_accuracies, val_accuracies, output_file=None):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch', fontweight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Accuracy', fontweight="bold", fontsize=14, labelpad=20)
    plt.title('Training and Validation Accuracy', fontweight="bold", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Accuracy curves saved to {output_file}")
    else:
        plt.savefig("Training_and_Validation_Accuracy.png", dpi=300)
    plt.close()

def plot_roc_curve(model, data, output_file=None):
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index, data.edge_attr)
        predictions = predictions.cpu().numpy()
        labels = data.y.cpu().numpy()
        train_mask = data.train_mask.cpu().numpy()
        val_mask = data.val_mask.cpu().numpy()
        test_mask = data.test_mask.cpu().numpy()
        plt.figure(figsize=(10, 8), dpi=300)
        if np.unique((labels[train_mask] >= 0.5).astype(int)).size > 1:
            train_fpr, train_tpr, _ = roc_curve((labels[train_mask] >= 0.5).astype(int), predictions[train_mask])
            train_auc = auc(train_fpr, train_tpr)
            plt.plot(train_fpr, train_tpr, label=f'Training (AUC = {train_auc:.3f})')
        if np.unique((labels[val_mask] >= 0.5).astype(int)).size > 1:
            val_fpr, val_tpr, _ = roc_curve((labels[val_mask] >= 0.5).astype(int), predictions[val_mask])
            val_auc = auc(val_fpr, val_tpr)
            plt.plot(val_fpr, val_tpr, label=f'Validation (AUC = {val_auc:.3f})')
        if np.unique((labels[test_mask] >= 0.5).astype(int)).size > 1:
            test_fpr, test_tpr, _ = roc_curve((labels[test_mask] >= 0.5).astype(int), predictions[test_mask])
            test_auc = auc(test_fpr, test_tpr)
            plt.plot(test_fpr, test_tpr, label=f'Test (AUC = {test_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate', fontweight="bold", fontsize=14, labelpad=20)
        plt.ylabel('True Positive Rate', fontweight="bold", fontsize=14, labelpad=20)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight="bold", fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        if output_file:
            plt.savefig(output_file, dpi=300)
            print(f"ROC curve saved to {output_file}")
        plt.show()

def plot_precision_recall_curve(model, data, output_file=None):
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index, data.edge_attr)
        predictions = predictions.cpu().numpy()
        labels = data.y.cpu().numpy()
        train_mask = data.train_mask.cpu().numpy()
        val_mask = data.val_mask.cpu().numpy()
        test_mask = data.test_mask.cpu().numpy()
        plt.figure(figsize=(10, 8), dpi=300)
        if np.unique((labels[train_mask] >= 0.5).astype(int)).size > 1:
            train_precision, train_recall, _ = precision_recall_curve((labels[train_mask] >= 0.5).astype(int), predictions[train_mask])
            train_ap = average_precision_score((labels[train_mask] >= 0.5).astype(int), predictions[train_mask])
            plt.plot(train_recall, train_precision, label=f'Training (AP = {train_ap:.3f})')
        if np.unique((labels[val_mask] >= 0.5).astype(int)).size > 1:
            val_precision, val_recall, _ = precision_recall_curve((labels[val_mask] >= 0.5).astype(int), predictions[val_mask])
            val_ap = average_precision_score((labels[val_mask] >= 0.5).astype(int), predictions[val_mask])
            plt.plot(val_recall, val_precision, label=f'Validation (AP = {val_ap:.3f})')
        if np.unique((labels[test_mask] >= 0.5).astype(int)).size > 1:
            test_precision, test_recall, _ = precision_recall_curve((labels[test_mask] >= 0.5).astype(int), predictions[test_mask])
            test_ap = average_precision_score((labels[test_mask] >= 0.5).astype(int), predictions[test_mask])
            plt.plot(test_recall, test_precision, label=f'Test (AP = {test_ap:.3f})')
        plt.xlabel('Recall', fontweight="bold", fontsize=14, labelpad=20)
        plt.ylabel('Precision', fontweight="bold", fontsize=14, labelpad=20)
        plt.title('Precision-Recall Curve', fontweight="bold", fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        if output_file:
            plt.savefig(output_file, dpi=300)
            print(f"PR curve saved to {output_file}")
        plt.show()

def feature_importance_analysis(model, feature_names):
    weights = model.conv1.lin.weight.detach().cpu().numpy()
    importance_scores = np.abs(weights).sum(axis=0)
    importance_scores = importance_scores / importance_scores.sum()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    return importance_df

def plot_feature_importance(importance_df, output_file=None):
    plt.figure(figsize=(12, 8), dpi=300)
    importance_df = importance_df.sort_values('Importance')
    bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', ha='left', va='center', fontweight='bold')
    plt.xlabel('Importance Score', fontweight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Feature', fontweight="bold", fontsize=14, labelpad=20)
    plt.title('Feature Importance Analysis', fontweight="bold", fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Feature importance plot saved to {output_file}")
    else:
        plt.savefig("Feature_Importance.png", dpi=300)
    plt.close()

def plot_error_distribution(predictions, true_values, set_name="Train", output_file=None):
    errors = predictions - true_values
    plt.figure(figsize=(10, 6), dpi=300)
    plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2, label=f'Mean Error: {np.mean(errors):.4f}')
    plt.xlabel('Prediction Error', fontweight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Frequency', fontweight="bold", fontsize=14, labelpad=20)
    plt.title(f'{set_name} Set Error Distribution', fontweight="bold", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"{set_name} error distribution saved to {output_file}")
    else:
        plt.savefig(f"{set_name}_Error_Distribution.png", dpi=300)
    plt.close()

def plot_predictions_vs_true(predictions, true_values, set_name="Train", output_file=None):
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(true_values, predictions, alpha=0.6, edgecolors='w', s=100)
    min_val = min(np.min(true_values), np.min(predictions))
    max_val = max(np.max(true_values), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    r2 = r2_score(true_values, predictions) if len(true_values) > 1 else float('nan')
    plt.xlabel('True Values', fontweight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Predicted Values', fontweight="bold", fontsize=14, labelpad=20)
    plt.title(f'{set_name} Set: Predictions vs True Values (R² = {r2:.4f})', fontweight="bold", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"{set_name} predictions vs true values plot saved to {output_file}")
    else:
        plt.savefig(f"{set_name}_Predictions_vs_True.png", dpi=300)
    plt.close()

def plot_confusion_matrix(conf_matrix, set_name="Train", output_file=None):
    plt.figure(figsize=(8, 6), dpi=300)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{set_name} Set Confusion Matrix', fontweight="bold", fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], fontweight="bold")
    plt.yticks(tick_marks, ['Negative', 'Positive'], fontweight="bold")
    plt.xlabel('Predicted Label', fontweight="bold", fontsize=14, labelpad=20)
    plt.ylabel('True Label', fontweight="bold", fontsize=14, labelpad=20)
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'), ha="center", va="center", color="white" if conf_matrix[i, j] > thresh else "black", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"{set_name} confusion matrix saved to {output_file}")
    else:
        plt.savefig(f"{set_name}_Confusion_Matrix.png", dpi=300)
    plt.close()

def plot_metrics_over_epochs(epoch_metrics, output_file=None):
    metrics_to_plot = [
        ('Accuracy', ['train_acc', 'val_acc']),
        ('Loss', ['train_loss', 'val_loss']),
        ('Precision', ['train_precision', 'val_precision']),
        ('Recall', ['train_recall', 'val_recall']),
        ('F1 Score', ['train_f1', 'val_f1']),
        ('R²', ['train_r2', 'val_r2'])
    ]
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics), dpi=300)
    for i, (metric_name, metric_keys) in enumerate(metrics_to_plot):
        ax = axes[i]
        ax.plot(epoch_metrics[metric_keys[0]], label=f'Training {metric_name}')
        ax.plot(epoch_metrics[metric_keys[1]], label=f'Validation {metric_name}')
        ax.set_xlabel('Epoch', fontweight="bold", fontsize=14, labelpad=20)
        ax.set_ylabel(metric_name, fontweight="bold", fontsize=14, labelpad=20)
        ax.set_title(f'Training and Validation {metric_name} Over Epochs', fontweight="bold", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Metrics over epochs plot saved to {output_file}")
    else:
        plt.savefig("Metrics_Over_Epochs.png", dpi=300)
    plt.close()

def plot_distribution_of_scores(predictions_df, output_file=None):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.hist(predictions_df['PredictedScore'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    mean_score = predictions_df['PredictedScore'].mean()
    plt.axvline(x=mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean Score: {mean_score:.4f}')
    median_score = predictions_df['PredictedScore'].median()
    plt.axvline(x=median_score, color='green', linestyle='-.', linewidth=2, label=f'Median Score: {median_score:.4f}')
    plt.xlabel('Predicted Score', fontweight="bold", fontsize=14, labelpad=20)
    plt.ylabel('Frequency', fontweight="bold", fontsize=14, labelpad=20)
    plt.title('Distribution of Predicted Therapeutic Target Scores', fontweight="bold", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Distribution of scores plot saved to {output_file}")
    else:
        plt.savefig("Score_Distribution.png", dpi=300)
    plt.close()

def nested_cross_validation_pipeline(features_df, G, outer_folds=5, inner_folds=5, max_epochs=200, output_dir="results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    feature_columns = ['Log2FC', 'Degree', 'Betweenness', 'Closeness', 'AvgCombinedScore']
    if 'InPathway' in features_df.columns:
        feature_columns.append('InPathway')
    if 'ImportanceScore' not in features_df.columns:
        formula_components = []
        formula_components.append('features_df["Log2FC"].abs() * 0.15')
        formula_components.append('features_df["Degree"] * 0.15')
        formula_components.append('features_df["Betweenness"] * 0.15')
        formula_components.append('features_df["Closeness"] * 0.15')
        formula_components.append('features_df["AvgCombinedScore"] * 0.2')
        if 'InPathway' in features_df.columns:
            formula_components.append('features_df["InPathway"] * 0.2')
        importance_formula = ' + '.join(formula_components)
        features_df['ImportanceScore'] = eval(importance_formula)
        scaler = StandardScaler()
        features_df['ImportanceScore'] = scaler.fit_transform(features_df[['ImportanceScore']])
        features_df['ImportanceScore'] = 1 / (1 + np.exp(-features_df['ImportanceScore']))
    n_samples = len(features_df)
    indices = np.arange(n_samples)
    outer_kf = KFold(n_splits=outer_folds, shuffle=True, random_state=RANDOM_SEED)
    outer_fold_results = []
    all_predictions = []
    fold_no = 0
    for outer_train_idx, outer_test_idx in outer_kf.split(indices):
        fold_no += 1
        print(f"\nOuter fold {fold_no}/{outer_folds} - Train size: {len(outer_train_idx)}, Test size: {len(outer_test_idx)}")
        inner_kf = KFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_SEED)
        val_losses_per_epoch = np.zeros((inner_folds, max_epochs))
        inner_fold = 0
        best_epoch_per_inner = []
        start_inner = time.time()
        for train_idx_inner, val_idx_inner in inner_kf.split(outer_train_idx):
            inner_fold += 1
            train_idx = outer_train_idx[train_idx_inner]
            val_idx = outer_train_idx[val_idx_inner]
            data_inner = build_data_object_from_features(features_df, G, train_idx=train_idx, val_idx=val_idx, test_idx=None)
            feature_dim = len(feature_columns)
            model_inner = TargetIdentificationModel(feature_dim=feature_dim)
            optimizer_inner = torch.optim.Adam(model_inner.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()
            best_val_loss = float('inf')
            best_epoch = 0
            for epoch in range(max_epochs):
                model_inner.train()
                optimizer_inner.zero_grad()
                out = model_inner(data_inner.x, data_inner.edge_index, data_inner.edge_attr)
                loss = loss_fn(out[data_inner.train_mask], data_inner.y[data_inner.train_mask])
                loss.backward()
                optimizer_inner.step()
                model_inner.eval()
                with torch.no_grad():
                    out = model_inner(data_inner.x, data_inner.edge_index, data_inner.edge_attr)
                    val_preds = out[data_inner.val_mask]
                    val_loss = loss_fn(val_preds, data_inner.y[data_inner.val_mask]).item()
                val_losses_per_epoch[inner_fold-1, epoch] = val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
            best_epoch_per_inner.append(best_epoch)
            print(f"  Inner fold {inner_fold}: best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")
        end_inner = time.time()
        print(f"  Inner CV time: {end_inner-start_inner:.1f}s")
        mean_val_loss_by_epoch = val_losses_per_epoch.mean(axis=0)
        chosen_epoch = int(np.argmin(mean_val_loss_by_epoch)) + 1
        print(f"  Chosen number of epochs after inner CV: {chosen_epoch}")
        data_outer = build_data_object_from_features(features_df, G, train_idx=outer_train_idx, val_idx=None, test_idx=outer_test_idx)
        train_sub_idx, val_sub_idx = train_test_split(outer_train_idx, train_size=0.85, random_state=RANDOM_SEED)
        data_outer = build_data_object_from_features(features_df, G, train_idx=train_sub_idx, val_idx=val_sub_idx, test_idx=outer_test_idx)
        feature_dim = len(feature_columns)
        final_model = TargetIdentificationModel(feature_dim=feature_dim)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        final_model, tr_losses, va_losses, epoch_metrics, _ = train_single_run(final_model, data_outer, optimizer, loss_fn, num_epochs=chosen_epoch, verbose=False)
        metrics = evaluate_model_numeric(final_model, data_outer)
        print(f"Outer fold {fold_no} metrics: test_mse={metrics['test_mse']:.4f}, test_rmse={metrics['test_rmse']:.4f}, test_r2={metrics['test_r2']:.4f}, test_accuracy={metrics['test_accuracy']:.4f}")
        final_model.eval()
        with torch.no_grad():
            preds = final_model(data_outer.x, data_outer.edge_index, data_outer.edge_attr).cpu().numpy().reshape(-1,)
        df_preds = features_df[['Gene']].copy()
        df_preds['TrueScore'] = features_df['ImportanceScore'].values
        df_preds['PredictedScore'] = preds
        df_preds['OuterFold'] = fold_no
        df_preds['IsTest'] = False
        df_preds.loc[outer_test_idx, 'IsTest'] = True
        all_predictions.append(df_preds)
        fold_metrics = metrics.copy()
        fold_metrics['outer_fold'] = fold_no
        outer_fold_results.append(fold_metrics)
        model_file = os.path.join(output_dir, f"model_outer_fold_{fold_no}.pt")
        torch.save(final_model.state_dict(), model_file)
        print(f"Saved model for outer fold {fold_no} to {model_file}")
        preds_file = os.path.join(output_dir, f"predictions_outer_fold_{fold_no}.csv")
        df_preds.to_csv(preds_file, index=False)
        print(f"Saved predictions for outer fold {fold_no} to {preds_file}")
    results_df = pd.DataFrame(outer_fold_results)
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    summary = {}
    for col in numeric_cols:
        if col == 'outer_fold':
            continue
        summary[f"{col}_mean"] = results_df[col].mean()
        summary[f"{col}_sd"] = results_df[col].std()
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(output_dir, "nested_cv_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\nNested CV summary saved to {summary_file}")
    per_fold_file = os.path.join(output_dir, "nested_cv_per_fold_metrics.csv")
    results_df.to_csv(per_fold_file, index=False)
    print(f"Per-fold metrics saved to {per_fold_file}")
    all_preds_df = pd.concat(all_predictions, ignore_index=True)
    all_preds_file = os.path.join(output_dir, "nested_cv_all_predictions.csv")
    all_preds_df.to_csv(all_preds_file, index=False)
    print(f"All predictions saved to {all_preds_file}")
    return results_df, summary_df, all_preds_df

def identify_therapeutic_targets(exp_file, ppi_file, pat_file=None, top_n=10, output_dir="results"):
    print("Loading data...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gene_exp_df = load_gene_expression(exp_file)
    ppi_df = load_ppi_data(ppi_file)
    pathway_genes = None
    if pat_file and os.path.exists(pat_file):
        pathway_genes = load_pathway_genes(pat_file)
        print(f"Loaded {len(pathway_genes)} genes from pathway file")
    else:
        print("No pathway file provided or file not found. Proceeding without pathway data.")
    print("Creating features...")
    features_df, G = create_features(gene_exp_df, ppi_df, pathway_genes)
    print("Running nested cross-validation...")
    OUTER_FOLDS = 5
    INNER_FOLDS = 5
    MAX_EPOCHS = 200
    per_fold_results, summary_df, all_preds_df = nested_cross_validation_pipeline(features_df, G, outer_folds=OUTER_FOLDS, inner_folds=INNER_FOLDS, max_epochs=MAX_EPOCHS, output_dir=output_dir)
    feature_columns = ['Log2FC', 'Degree', 'Betweenness', 'Closeness', 'AvgCombinedScore']
    if 'InPathway' in features_df.columns:
        feature_columns.append('InPathway')
    n = len(features_df)
    all_indices = list(range(n))
    train_idx, val_idx = train_test_split(all_indices, train_size=0.85, random_state=RANDOM_SEED)
    data_all = build_data_object_from_features(features_df, G, train_idx=train_idx, val_idx=val_idx, test_idx=None)
    feature_dim = len(feature_columns)
    final_model = TargetIdentificationModel(feature_dim=feature_dim)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    final_model, train_losses, val_losses, epoch_metrics, _ = train_single_run(final_model, data_all, optimizer, loss_fn, num_epochs=100, verbose=True)
    final_model.eval()
    with torch.no_grad():
        preds_all = final_model(data_all.x, data_all.edge_index, data_all.edge_attr).cpu().numpy().reshape(-1,)
    features_df['PredictedScore'] = preds_all
    top_targets = rank_targets(features_df[['Gene','PredictedScore','Log2FC'] + (['InPathway'] if 'InPathway' in features_df.columns else [])], top_n)
    results_file = os.path.join(output_dir, "top_targets.csv")
    save_results(top_targets, results_file)
    network_file = os.path.join(output_dir, "network_visualization.png")
    visualize_network(G, top_targets, network_file)
    print("\nNested cross-validation completed. Summary:")
    print(summary_df.T)
    importance_df = feature_importance_analysis(final_model, feature_columns)
    importance_file = os.path.join(output_dir, "feature_importance.csv")
    importance_df.to_csv(importance_file, index=False)
    plot_feature_importance(importance_df, os.path.join(output_dir, "feature_importance.png"))
    plot_training_validation_curves(train_losses, val_losses, os.path.join(output_dir, "final_training_validation_loss.png"))
    plot_accuracy_curves(epoch_metrics['train_acc'], epoch_metrics['val_acc'], os.path.join(output_dir, "final_training_validation_accuracy.png"))
    plot_distribution_of_scores(features_df, os.path.join(output_dir, "final_score_distribution.png"))
    return top_targets, final_model, G, per_fold_results, summary_df

def create_example_csv_files():
    if not os.path.exists(EXPRESSION_FILE):
        exp_data = """Target
KRT6A
KRT17
KRT5
SFN
MPO"""
        with open(EXPRESSION_FILE, 'w') as f:
            f.write(exp_data)
        print(f"Created example gene expression file (single column): {EXPRESSION_FILE}")
    if not os.path.exists(PPI_FILE):
        ppi_data = """node1,node2,neighborhood_on_chromosome,gene_fusion,phylogenetic_cooccurrence,homology,coexpression,experimentally_determined_interaction,database_annotated,automated_textmining,combined_score
CD74,HLA-DRA,0,0,0,0,0.852,0.94,0.9,0.664,0.999
CLDN3,KRT5,0,0,0,0,0.067,0.071,0,0.456,0.487
HLA-DRA,CD74,0,0,0,0,0.852,0.94,0.9,0.664,0.999
KRT17,KRT5,0,0,0.104,0.722,0.653,0.166,0,0.546,0.866
SFN,S100A2,0,0,0,0,0.478,0.094,0,0.309,0.644"""
        with open(PPI_FILE, 'w') as f:
            f.write(ppi_data)
        print(f"Created example PPI file: {PPI_FILE}")
    if PATHWAY_FILE and not os.path.exists(PATHWAY_FILE):
        pathway_data = """Target
"""
        with open(PATHWAY_FILE, 'w') as f:
            f.write(pathway_data)
        print(f"Created example pathway file: {PATHWAY_FILE}")

if __name__ == "__main__":
    print("Therapeutic Target Identification with Nested CV")
    print("================================")
    create_example_csv_files()
    print(f"Expression data: {EXPRESSION_FILE}")
    print(f"PPI data: {PPI_FILE}")
    print(f"Pathway data: {PATHWAY_FILE if PATHWAY_FILE else 'Not provided'}")
    print(f"Top targets to identify: {TOP_N_TARGETS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("================================")
    top_targets, model, G, per_fold_results, summary_df = identify_therapeutic_targets(EXPRESSION_FILE, PPI_FILE, PATHWAY_FILE, TOP_N_TARGETS, OUTPUT_DIR)

    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR} directory.")
