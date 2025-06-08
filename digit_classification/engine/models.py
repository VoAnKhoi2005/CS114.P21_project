import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class ClassicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ClassicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransferLearningModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained=True, freeze_features=True, grayscale=False):
        super().__init__()
        self.model_name = model_name.lower()

        # Get weights enum dynamically if pretrained
        weights = self._get_weights_enum(self.model_name) if pretrained else None

        # Load model with weights param instead of deprecated pretrained
        self.model = getattr(models, self.model_name)(weights=weights)

        # Convert to grayscale input if requested
        if grayscale:
            self._convert_to_grayscale()

        # Freeze all layers if requested
        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace classifier head with new output layer
        self._replace_classifier(num_classes)

    def _get_weights_enum(self, model_name):
        name_camel = ''.join(word.capitalize() for word in model_name.split('_'))
        weights_enum_name = f"{name_camel}_Weights"

        weights_enum = getattr(models, weights_enum_name, None)
        if weights_enum is not None:
            # Return the default weights variant
            default_weights = getattr(weights_enum, "DEFAULT", None)
            if default_weights is not None:
                return default_weights

        # Fallback if no weights found
        return None

    def _convert_to_grayscale(self):
        # Commonly input conv layer is named 'conv1' in torchvision models like ResNet
        if hasattr(self.model, 'conv1'):
            old_conv = self.model.conv1
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias[:] = old_conv.bias
            self.model.conv1 = new_conv
        else:
            raise NotImplementedError(f"Grayscale input conversion not implemented for model {self.model_name}")

    def _replace_classifier(self, num_classes):
        # Try common classifier attribute names and replace the final linear layer
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Sequential):
                # Replace last layer in Sequential
                in_features = self.model.classifier[-1].in_features
                self.model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'head'):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_classes)
        else:
            raise NotImplementedError(f"Unknown classifier structure for model {self.model_name}")

    def forward(self, x):
        return self.model(x)