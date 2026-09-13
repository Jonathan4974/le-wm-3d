import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "baseline_ckpt", "hf_cube")
out = Path(swm.data.utils.get_cache_dir(), "baseline_ckpt", "cube", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
# model = JEPA(
#     encoder=encoder,
#     predictor=ARPredictor(**cfg["predictor"]),
#     action_encoder=Embedder(**cfg["action_encoder"]),
#     projector=mlp("projector"),
#     pred_proj=mlp("pred_proj"),
# )

# Process predictor configuration
predictor_params = cfg["predictor"].copy()
predictor_params.pop("_target_", None)  # Remove '_target_' if present to avoid unexpected argument errors

# Process action_encoder configuration (just in case it has the same issue)
action_params = cfg["action_encoder"].copy()
action_params.pop("_target_", None)

model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**predictor_params), # Use processed parameters
    action_encoder=Embedder(**action_params),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)

sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
