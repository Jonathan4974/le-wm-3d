from lightning.pytorch.callbacks import Callback
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients


class ModelDiagnosticsCallback(Callback):
    """
    Custom callback to monitor global weight/gradient stats 
    and BatchNorm moving averages in Weights & Biases.
    """
    def __init__(self):
        self.bn_stats = {}

    def on_fit_start(self, trainer, pl_module):

        def make_hook(name):
            def hook(module, inputs, outputs):

                x = inputs[0]

                if x.ndim == 2:
                    var_per_channel = x.var(dim=0,unbiased=False)
                elif x.ndim == 3:
                    var_per_channel = x.var(dim=(0, 2),unbiased=False)
                elif x.ndim == 4:
                    var_per_channel = x.var(dim=(0, 2, 3),unbiased=False)

                self.bn_stats[f"bn_stats/{name}.batch_var_min"] = var_per_channel.min().item()
                self.bn_stats[f"bn_stats/{name}.batch_var_max"] = var_per_channel.max().item()

            return hook

        for name, module in pl_module.named_modules():
            if isinstance(module,(nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
                module.register_forward_hook(make_hook(name))

    def on_after_backward(self, trainer, pl_module):
        # Align with the trainer's log_every_n_steps
        if (trainer.global_step + 1) % trainer.log_every_n_steps == 0:
            total_w_norm_sq = 0.0
            total_g_norm_sq = 0.0
            all_w_means = []
            all_g_means = []

            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    all_w_means.append(param.data.mean().item())
                    total_w_norm_sq += param.data.norm(2).item() ** 2
                    
                    if param.grad is not None:
                        all_g_means.append(param.grad.data.mean().item())
                        total_g_norm_sq += param.grad.data.norm(2).item() ** 2

            metrics = {
                "meta/weight_mean": sum(all_w_means) / len(all_w_means) if all_w_means else 0.0,
                "meta/weight_norm": total_w_norm_sq ** 0.5,
            }
            if all_g_means:
                metrics["meta/grad_mean"] = sum(all_g_means) / len(all_g_means)
                metrics["meta/grad_norm"] = total_g_norm_sq ** 0.5

            # print(f"\n--- DEBUG: weight norm {total_w_norm_sq} at step {trainer.global_step} ---")
            pl_module.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % trainer.log_every_n_steps == 0:
            bn_metrics = {}
            for name, module in pl_module.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    running_mean_val = module.running_mean.detach().mean().item() if module.running_mean is not None else 0.0
                    running_var_val = module.running_var.detach().mean().item() if module.running_var is not None else 1.0
                    
                    bn_metrics[f"bn_stats/{name}.running_mean"] = running_mean_val
                    bn_metrics[f"bn_stats/{name}.running_var"] = running_var_val
                    
                    bn_metrics[f"bn_stats/{name}.running_var_min"] = module.running_var.detach().min().item()
                    bn_metrics[f"bn_stats/{name}.running_var_max"] = module.running_var.detach().max().item()
                    bn_metrics[f"bn_stats/{name}.running_mean_absmax"] = module.running_mean.abs().max().item()

                    if module.weight is not None:
                        bn_metrics[f"bn_stats/{name}.gamma_mean"] = module.weight.mean().item()
                        bn_metrics[f"bn_stats/{name}.gamma_absmax"] = module.weight.abs().max().item()

                    if module.bias is not None:
                        bn_metrics[f"bn_stats/{name}.beta_absmax"] = module.bias.abs().max().item()
                    

            print(f"\n=== Batch {batch_idx} ===")
            print("logger type:", type(trainer.logger))
            print("global_step:", trainer.global_step)
            print("num bn metrics:", len(bn_metrics))

            if hasattr(self, "bn_stats"):
                bn_metrics.update(self.bn_stats)

            # print(f"\n--- DEBUG: Found {len(bn_metrics)} BN metrics at step {trainer.global_step} ---")
            if bn_metrics:
                print("First few keys:", list(bn_metrics.keys())[:2])
                pl_module.log_dict(bn_metrics, on_step=True, on_epoch=False, prog_bar=False)



class PredictorFeatureImportanceWrapper(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        
    def forward(self, ctx_emb, ctx_act):
        """
        ctx_emb:  [B, ctx_len, D] -> ViT patch context embeddings
        ctx_act:  [B, ctx_len, Act_D] -> Action context embeddings
        """
        # 1. Run the predictor forward pass
        pred_emb = self.predictor(ctx_emb, ctx_act) # Shape: [B, n_preds, D]
        
        # 2. Convert the entire output tensor into a scalar energy value.
        # We use a Frobenius-like norm (sum of squares).
        # A change in an input feature that vastly changes this sum has high importance.
        scalar_energy = torch.sum(pred_emb ** 2).view(1)
        
        return scalar_energy
    

class LeWMPredictorIGCallback(Callback):
    def __init__(self, every_n_epochs: int = 1, n_steps: int = 30, ctx_len: int = 1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.n_steps = n_steps
        self.ctx_len = ctx_len
        self.ig = None
        self.wrapper = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx != 0 or (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        if self.ig is None:
            self.wrapper = PredictorFeatureImportanceWrapper(pl_module.model.predict)
            self.ig = IntegratedGradients(self.wrapper)

        # Clean NaNs just like the main forward pass
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        with torch.enable_grad():
            # Run the visual and action encoders
            output = pl_module.model.encode(batch)
            emb = output["emb"]
            act_emb = output["act_emb"]

            # Isolate and clone the exact context slices
            ctx_emb = emb[:, :self.ctx_len].detach().clone().requires_grad_(True)
            ctx_act = act_emb[:, :self.ctx_len].detach().clone().requires_grad_(True)

            # Baselines
            ctx_emb_baseline = torch.zeros_like(ctx_emb)
            ctx_act_baseline = torch.zeros_like(ctx_act)

            # Compute Attributions
            attributions = self.ig.attribute(
                inputs=(ctx_emb, ctx_act),
                baselines=(ctx_emb_baseline, ctx_act_baseline),
                n_steps=self.n_steps
            )
            
            emb_attr, act_attr = attributions

        # ---- Aggregate Metrics over the context window ----
        # Calculate overall visual feature vs action feature importance
        vit_importance = torch.abs(emb_attr).sum().item()
        action_importance = torch.abs(act_attr).sum().item()
        total = vit_importance + action_importance + 1e-8
        
        vit_pct = (vit_importance / total) * 100
        action_pct = (action_importance / total) * 100

        logger = trainer.logger
        if logger:
            # Bypasses lightning's batch averaging and logs raw step values
            logger.log_metrics(
                metrics={
                    "predictor/vit_importance_pct": vit_pct,
                    "predictor/action_importance_pct": action_pct,
                },
                step=trainer.global_step
            )


