"""
Early Stopping and Overfitting Detection for SCI Training

Required by: SCI_ENGINEERING_STANDARDS.md Section 5.3
"""


class EarlyStopping:
    """Early stopping with patience and overfitting detection."""

    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        """
        Args:
            patience: LOW #84: Number of epochs to wait before stopping (default: 5)
                     - Typical values: 3-10 for standard training
                     - Higher values (10+) for noisy metrics or slow convergence
                     - Lower values (3-5) for fast-converging models
                     - For SCI on SCAN: 5 is recommended
            min_delta: Minimum improvement to count as progress
            mode: 'max' for metrics where higher is better (accuracy),
                  'min' for metrics where lower is better (loss)
        """
        # HIGH #44: Validate mode parameter
        assert mode in ['max', 'min'], \
            f"mode must be 'max' or 'min', got '{mode}'"

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        """
        Check if should stop training.

        Args:
            score: Current metric value
            epoch: Current epoch number

        Returns:
            bool: True if should stop training
        """

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class OverfittingDetector:
    """Detects overfitting by comparing train/val losses."""

    def __init__(self,
                 threshold_ratio=1.5,  # Val loss > 1.5x train loss
                 window_size=3,  # Average over 3 epochs
                 min_epochs=5):  # Don't detect before epoch 5
        """
        Args:
            threshold_ratio: Ratio threshold (val_loss / train_loss) for overfitting
            window_size: Number of epochs to average over
            min_epochs: Minimum epochs before detection starts
        """
        self.threshold_ratio = threshold_ratio
        self.window_size = window_size
        self.min_epochs = min_epochs
        self.train_losses = []
        self.val_losses = []

    def update(self, train_loss, val_loss, epoch):
        """
        Update with new losses and check for overfitting.

        Args:
            train_loss: Training loss for current epoch
            val_loss: Validation loss for current epoch
            epoch: Current epoch number

        Returns:
            tuple: (is_overfitting: bool, ratio: float or None)
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if epoch < self.min_epochs:
            return False, None

        if len(self.train_losses) < self.window_size:
            return False, None

        # Compute windowed averages
        recent_train = sum(self.train_losses[-self.window_size:]) / self.window_size
        recent_val = sum(self.val_losses[-self.window_size:]) / self.window_size

        # Check ratio
        ratio = recent_val / (recent_train + 1e-8)

        is_overfitting = ratio > self.threshold_ratio

        return is_overfitting, ratio
