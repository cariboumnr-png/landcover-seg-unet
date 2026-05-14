import torch
from typing import List, Tuple, Optional

def _compute_cm_tensor(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Computes a dense confusion matrix tensor using fast 1D bincount operations.
    """
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)
    
    if preds_flat.shape != labels_flat.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape} and labels {labels.shape} must have the same number of elements.")
        
    # Flat indexing: (True * num_classes) + Pred
    indices = (num_classes * labels_flat) + preds_flat
    
    # Calculate bincount and reshape to 2D
    cm_tensor = torch.bincount(indices, minlength=num_classes**2)
    return cm_tensor.reshape(num_classes, num_classes)

def _format_cm_markdown(cm_tensor: torch.Tensor, class_names: List[str]) -> str:
    """
    Formats a 2D confusion matrix tensor into a TensorBoard-compatible Markdown table.
    """
    num_classes = len(class_names)
    
    # Table headers
    header = "| True \\ Pred | " + " | ".join(class_names) + " |"
    separator = "|" + "|".join(["---"] * (num_classes + 1)) + "|"
    
    rows = [header, separator]
    
    # Table rows
    for i in range(num_classes):
        row_data = [f"**{class_names[i]}**"] 
        for j in range(num_classes):
            row_data.append(str(cm_tensor[i, j].item()))
        
        rows.append("| " + " | ".join(row_data) + " |")
        
    return "\n".join(rows)

def get_confusion_matrix_and_table(
    preds: torch.Tensor, 
    labels: torch.Tensor, 
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None
) -> Tuple[torch.Tensor, str]:
    """
    Generates a confusion matrix tensor and a Markdown table for TensorBoard logging.
    
    Args:
        preds (torch.Tensor): Tensor of integer predictions.
        labels (torch.Tensor): Tensor of integer ground truth labels.
        class_names (List[str], optional): List of string names for each class. 
            If None, defaults to ["Class 0", "Class 1", ...].
        num_classes (int, optional): Total number of classes. If None, inferred 
            from the maximum value in preds and labels.
            
    Returns:
        Tuple[torch.Tensor, str]: A tuple containing:
            - A 2D PyTorch tensor representing the confusion matrix.
            - A Markdown-formatted string representing the confusion matrix table.
    """
    # 1. Infer num_classes if not provided
    if num_classes is None:
        if class_names is not None:
            num_classes = len(class_names)
        else:
            num_classes = max(preds.max().item(), labels.max().item()) + 1
            
    # 2. Generate default class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
        
    # 3. Validate class_names length matches num_classes
    if len(class_names) != num_classes:
        raise ValueError(f"Length of class_names ({len(class_names)}) must match num_classes ({num_classes}).")
        
    # 4. Compute outputs
    cm_tensor = _compute_cm_tensor(preds, labels, num_classes)
    cm_markdown = _format_cm_markdown(cm_tensor, class_names)
    
    return cm_tensor, cm_markdown

# =====================================================================
# Example usage block (can be removed when importing into your project)
# =====================================================================
if __name__ == "__main__":
    # Dummy outputs from a network
    mock_labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    mock_preds = torch.tensor([0, 1, 1, 0, 0, 2, 2])
    names = ["Cat", "Dog", "Bird"]
    
    cm, md_table = get_confusion_matrix_and_table(mock_preds, mock_labels, class_names=names)
    
    print("Tensor output:")
    print(cm)
    print("\nMarkdown output:")
    print(md_table)
