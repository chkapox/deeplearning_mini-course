import torch
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).
        """
        # Перенос данных на устройство и батч-трансформы
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        # Выбор набора метрик: train / inference
        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        # AMP включаем только если есть CUDA и инициализирован scaler
        use_amp = (
            getattr(self, "scaler", None) is not None
            and self.device.type == "cuda"
        )

        # ─── forward + loss под autocast ─────────────────────
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        # ─── backward / step ─────────────────────────────────
        if self.is_train:
            if use_amp:
                self.scaler.scale(batch["loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                self._clip_grad_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                batch["loss"].backward()
                self._clip_grad_norm()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # ─── metrics update ──────────────────────────────────
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch
