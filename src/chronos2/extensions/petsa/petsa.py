import math
import warnings
from typing import Optional, Sequence, Mapping, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from chronos2.model import Chronos2Model
from chronos2.layers import TimeSelfAttention
from chronos2.pipeline import Chronos2Pipeline
from chronos2.dataset import Chronos2Dataset, DatasetMode, TensorOrArray


class BatchedLoRALayer(nn.Module):
    """
    Wraps a linear layer to add Batched Low-Rank Adaptation (LoRA).
    Allows applying a DIFFERENT LoRA adapter to every sample in the batch.
    
    W'[b] = W + alpha * (B[b] @ A[b])
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.in_dim = original_layer.in_features
        self.out_dim = original_layer.out_features
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # These will be initialized when setup_batch is called
        # Shapes: (batch_size, rank, in_dim) and (batch_size, out_dim, rank)
        self.batched_lora_A = None
        self.batched_lora_B = None

    def setup_batch(self, batch_size: int):
        """
        Initializes LoRA matrices for a specific batch size.
        Must be called before forward pass.
        """
        dtype = self.original_layer.weight.dtype
        device = self.original_layer.weight.device
        
        # Initialize A: (batch, rank, in)
        self.batched_lora_A = nn.Parameter(
            torch.empty(batch_size, self.rank, self.in_dim, dtype=dtype, device=device)
        )
        # Initialize B: (batch, out, rank)
        self.batched_lora_B = nn.Parameter(
            torch.zeros(batch_size, self.out_dim, self.rank, dtype=dtype, device=device)
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.batched_lora_A is None:
            return
            
        # Kaiming Uniform for A (vectorized initialization is tricky, looping for correctness/simplicity)
        nn.init.kaiming_uniform_(self.batched_lora_A, a=math.sqrt(5))
            
        # Zeros for B
        nn.init.zeros_(self.batched_lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, in_dim)
        
        # Original output
        original_out = self.original_layer(x)
        
        if self.batched_lora_A is None:
            return original_out
            
        # Batched LoRA path
        # x: (B, L, D_in)
        # A: (B, R, D_in) -> A.mT: (B, D_in, R)
        # B: (B, D_out, R) -> B.mT: (B, R, D_out)
        
        # Step 1: x @ A.T  -> (B, L, D_in) @ (B, D_in, R) = (B, L, R)
        temp = torch.bmm(x, self.batched_lora_A.transpose(1, 2))
        
        # Step 2: temp @ B.T -> (B, L, R) @ (B, R, D_out) = (B, L, D_out)
        lora_out = torch.bmm(temp, self.batched_lora_B.transpose(1, 2))
        
        return original_out + (lora_out * self.scaling)


class ChronosPETSAWrapper(nn.Module):
    """
    Wraps a Chronos2Model to implement Parameter-Efficient Test-Time Adaptation (PETSA)
    with BATCHED optimization.
    """
    def __init__(self, model: Chronos2Model, lora_rank: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.model = model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_layers = []
        
        # Freeze the entire base model first
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Inject LoRA layers
        self._inject_lora()
        
    def _inject_lora(self):
        """
        Iterate through the model and replace q, k, v projections in TimeSelfAttention
        with BatchedLoRALayer wrappers.
        """
        self.lora_layers = []
        # Chronos2EncoderBlock contains TimeSelfAttention at layer[0]
        for block in self.model.encoder.block:
            if isinstance(block.layer[0], TimeSelfAttention):
                time_attn = block.layer[0].self_attention
                
                # Wrap q, k, v
                time_attn.q = BatchedLoRALayer(time_attn.q, rank=self.lora_rank, alpha=self.lora_alpha)
                time_attn.k = BatchedLoRALayer(time_attn.k, rank=self.lora_rank, alpha=self.lora_alpha)
                time_attn.v = BatchedLoRALayer(time_attn.v, rank=self.lora_rank, alpha=self.lora_alpha)
                
                self.lora_layers.extend([time_attn.q, time_attn.k, time_attn.v])
                
    def setup_batch(self, batch_size: int):
        """Prepares all LoRA layers for a specific batch size"""
        for layer in self.lora_layers:
            layer.setup_batch(batch_size)

    def adapt_and_forecast(
        self,
        context: torch.Tensor,
        prediction_length: int,
        n_gradient_steps: int = 5,
        learning_rate: float = 1e-3,
        mask_ratio: float = 0.25,
        context_mask: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None,
        future_covariates_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs Batched PETSA.
        """
        self.model.eval() 
        batch_size, context_len = context.shape
        
        # 1. Initialize Batched Parameters
        self.setup_batch(batch_size)
        
        # 2. Collect all trainable params (the newly created batched params)
        # Note: We must locate them dynamically because setup_batch created new Tensors
        lora_params = []
        for layer in self.lora_layers:
            lora_params.append(layer.batched_lora_A)
            lora_params.append(layer.batched_lora_B)
            
        optimizer = AdamW(lora_params, lr=learning_rate)
        
        # --- Step 1: Self-Supervised Masking ---
        mask_len = int(context_len * mask_ratio)
        if mask_len < 1: mask_len = 1
            
        adaptation_context = context[:, :-mask_len]
        adaptation_target = context[:, -mask_len:]
        
        if context_mask is not None:
            adaptation_context_mask = context_mask[:, :-mask_len]
            adaptation_target_mask = context_mask[:, -mask_len:]
        else:
            adaptation_context_mask = None
            adaptation_target_mask = None
            
        patch_size = self.model.chronos_config.output_patch_size
        num_adaptation_patches = math.ceil(mask_len / patch_size)
        
        # --- Step 2: Adaptation Loop (Vectorized!) ---
        with torch.enable_grad():
            for _ in range(n_gradient_steps):
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(
                    context=adaptation_context,
                    context_mask=adaptation_context_mask,
                    future_target=adaptation_target,
                    future_target_mask=adaptation_target_mask,
                    num_output_patches=num_adaptation_patches
                )
                
                loss = output.loss
                
                loss.backward()
                optimizer.step()
            
        # --- Step 3: Forecasting ---
        with torch.no_grad():
            num_forecast_patches = math.ceil(prediction_length / patch_size)
            
            forecast_output = self.model(
                context=context,
                context_mask=context_mask,
                num_output_patches=num_forecast_patches,
                future_covariates=future_covariates,
                future_covariates_mask=future_covariates_mask
            )
            
            forecast = forecast_output.quantile_preds
            forecast = forecast[..., :prediction_length]
            
        # --- Step 4: Cleanup ---
        # No explicit cleanup needed, params are overwritten on next batch
        
        return forecast


class ChronosPETSAPipeline(Chronos2Pipeline):
    def __init__(self, model: ChronosPETSAWrapper):
        super().__init__(model=model.model)
        self.wrapper = model
        
    def predict(
        self,
        inputs: TensorOrArray
        | Sequence[TensorOrArray]
        | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]]],
        prediction_length: int | None = None,
        batch_size: int = 32, # Defaults to standard batch size now
        context_length: int | None = None,
        predict_batches_jointly: bool = False, # Unused
        limit_prediction_length: bool = False,
        **kwargs,
    ) -> list[torch.Tensor]:
        """
        Generate forecasts for the given time series using Batched PETSA.
        """
        model_prediction_length = self.model_prediction_length
        if prediction_length is None:
            prediction_length = model_prediction_length

        max_output_patches = kwargs.pop("max_output_patches", self.max_output_patches)
        unrolled_quantiles = kwargs.pop("unrolled_quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        # Validate kwargs
        if len(kwargs) > 0:
             # Just warn or ignore to be safe
             pass

        if context_length is None:
            context_length = self.model_context_length

        test_dataset = Chronos2Dataset.convert_inputs(
            inputs=inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=self.model_output_patch_size,
            mode=DatasetMode.TEST,
        )
        test_loader = DataLoader(test_dataset, batch_size=None, pin_memory=True, shuffle=False, drop_last=False)

        all_predictions: list[torch.Tensor] = []
        # Get model device from the model
        device = self.model.device
        
        for batch in test_loader:
            batch_context = batch["context"].to(device)
            batch_future_covariates = batch["future_covariates"]
            if batch_future_covariates is not None:
                batch_future_covariates = batch_future_covariates.to(device)
            
            # Use the Wrapper to run the full PETSA cycle on this batch
            batch_prediction = self.wrapper.adapt_and_forecast(
                context=batch_context,
                prediction_length=prediction_length,
                future_covariates=batch_future_covariates,
            )
            
            # Extend results (move back to CPU to save GPU memory)
            all_predictions.extend([ts_pred.cpu() for ts_pred in batch_prediction])

        return all_predictions

    def _predict_step(self, *args, **kwargs):
        pass
