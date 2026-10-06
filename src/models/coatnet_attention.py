import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import timm
except ImportError:
    timm = None

from src.config import TARGET_COLS

NUM_CLASSES = len(TARGET_COLS)  # 12


class FindingSpecificAttentionPooling(nn.Module):
    """
    12-Head Attention Pooling layer:
    Calculates independent attention weights over N windows for each of the
    12 target findings, allowing the model to focus on finding-specific planes
    and localized slices (e.g. Sagittal for ACL, Coronal for MCL, Axial for PF OA).
    """

    def __init__(self, in_features: int, num_classes: int = NUM_CLASSES, hidden_dim: int = 128):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        # Attention projection for each finding: W * tanh(V * f)
        self.attention_v = nn.Linear(in_features, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, num_classes)

        # Final classification heads per finding
        self.classifiers = nn.ModuleList([
            nn.Linear(in_features, 1) for _ in range(num_classes)
        ])

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Tensor of shape (N_windows, in_features)
        Returns:
            logits: Tensor of shape (num_classes,)
            attention_weights: Tensor of shape (num_classes, N_windows)
        """
        # Step 1: Compute attention scores: (N_windows, hidden_dim) -> (N_windows, num_classes)
        att_scores = self.attention_w(torch.tanh(self.attention_v(features)))

        # Step 2: Softmax along windows dimension (dim=0): (num_classes, N_windows)
        att_weights = F.softmax(att_scores.transpose(0, 1), dim=-1)

        # Step 3: Weighted sum of features for each class: (num_classes, in_features)
        context = torch.matmul(att_weights, features)

        # Step 4: Classify each finding from its specialized context vector
        logits = torch.stack([
            self.classifiers[k](context[k]).squeeze(-1) for k in range(self.num_classes)
        ], dim=-1)  # (num_classes,)

        return logits, att_weights


class KneeStudyModel(nn.Module):
    """
    Multi-View 2.5D Knee Study Model with Plane Conditioning and
    12-Head Finding-Specific Attention Pooling.
    """

    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = False,
        num_classes: int = NUM_CLASSES,
        num_planes: int = 3,
        use_plane_embedding: bool = True,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for backbone creation. Install with 'pip install timm'.")

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.use_plane_embedding = use_plane_embedding

        # Feature extractor backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head, extract feature vector
        )

        in_features = self.backbone.num_features

        # Plane embedding (0: Sagittal, 1: Coronal, 2: Axial)
        if use_plane_embedding:
            self.plane_embedding = nn.Embedding(num_planes, in_features)

        # 12-Head Finding-Specific Attention Pooling
        self.pool = FindingSpecificAttentionPooling(
            in_features=in_features,
            num_classes=num_classes,
        )

    def extract_window_features(self, windows: torch.Tensor, planes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract features from (N_windows, 3, H, W) and apply plane embeddings."""
        feat = self.backbone(windows)  # (N_windows, in_features)

        if self.use_plane_embedding and planes is not None:
            plane_emb = self.plane_embedding(planes)  # (N_windows, in_features)
            feat = feat + plane_emb

        return feat

    def forward_single_study(
        self, windows: torch.Tensor, planes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single patient study."""
        features = self.extract_window_features(windows, planes)
        logits, att_weights = self.pool(features)
        return logits, att_weights

    def forward(
        self, study_images: Union[torch.Tensor, list], study_planes: Optional[Union[torch.Tensor, list]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Batch-compatible forward pass:
        Handles list of study tensors [ (N1, 3, H, W), (N2, 3, H, W), ... ]
        """
        if isinstance(study_images, list):
            batch_logits = []
            batch_att = []
            for i, img in enumerate(study_images):
                p = study_planes[i] if study_planes is not None else None
                logits, att = self.forward_single_study(img, p)
                batch_logits.append(logits)
                batch_att.append(att)

            logits_tensor = torch.stack(batch_logits, dim=0)  # (B, 12)
            probs_tensor = torch.sigmoid(logits_tensor)        # (B, 12)
            return {"logits": logits_tensor, "probs": probs_tensor, "attention": batch_att}
        else:
            # Single study tensor passed directly
            logits, att = self.forward_single_study(study_images, study_planes)
            probs = torch.sigmoid(logits)
            return {"logits": logits.unsqueeze(0), "probs": probs.unsqueeze(0), "attention": [att]}


def create_model(
    backbone_name: str = "resnet34",
    checkpoint_path: Optional[Union[str, Path]] = None,
    pretrained: bool = False,
    device: str = "cpu",
) -> KneeStudyModel:
    """Factory helper to build and optionally load model weights."""
    model = KneeStudyModel(backbone_name=backbone_name, pretrained=pretrained)

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        print(f"Loading checkpoint weights from: {checkpoint_path}")
        state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        if "model" in state:
            model.load_state_dict(state["model"], strict=False)
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    model.to(device)
    return model


if __name__ == "__main__":
    print("Testing KneeStudyModel architecture...")
    device = "cpu"
    # Test with ResNet-34 control backbone
    model = create_model(backbone_name="resnet34", pretrained=False, device=device)
    model.eval()

    # Synthetic patient study: 16 windows of 2.5D RGB slices
    dummy_windows = torch.randn(16, 3, 384, 384, device=device)
    dummy_planes = torch.tensor([0, 0, 1, 1, 2, 2] * 3, device=device)[:16]

    with torch.no_grad():
        output = model(dummy_windows, dummy_planes)

    print(f"Output Logits Shape      : {output['logits'].shape}")
    print(f"Output Probabilities Shape: {output['probs'].shape}")
    print(f"Probabilities Preview     :\n{output['probs'].cpu().numpy().round(4)}")
    print(f"Attention Weights Shape   : {output['attention'][0].shape}")
    print("Model architecture verification successful!")
