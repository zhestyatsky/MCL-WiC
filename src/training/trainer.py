from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from src.data.constants import EPOCHS


def get_trainer(on_colab=True):
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=2,
        verbose=True,
        mode="min"
    )

    trainer = Trainer(
        gpus=1 if on_colab else None,
        checkpoint_callback=False,
        accumulate_grad_batches=10,
        max_epochs=EPOCHS,
        callbacks=[early_stop_callback],
        val_check_interval=0.5)

    return trainer
