# Runpod Worker

This worker hosts VideoCADFormer inference for the local `videocad-onshape` CLI.

## Required Runpod env vars

- `VIDEOCAD_RUNPOD_CHECKPOINT_PATH`
  - Container-local path for the model checkpoint.
  - Default: `/models/videocadformer.pt`
- `VIDEOCAD_RUNPOD_CHECKPOINT_GDRIVE_ID`
  - Optional Google Drive file id for auto-downloading the checkpoint on startup.
  - For the released checkpoint, use `1dWa8gZ56tXCLXzhuTTKjhq2bteUDf1ZH`.
- `VIDEOCAD_RUNPOD_MODEL_NAME`
  - Default: `cad_past_10_actions_and_states_timestep_embedding`
- `VIDEOCAD_RUNPOD_MODEL_CONFIG_PATH`
  - Default: `/app/src/videocad_onshape/vendor/model_configs/transformer_experiments.json`
- `VIDEOCAD_RUNPOD_DEVICE`
  - Usually `cuda` on a GPU worker.

## Request shape

The local client posts to:

`POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync`

With body:

```json
{
  "input": {
    "frames": ["<base64 png>", "..."],
    "action_history": [[0, 515, 500, -1, -1, -1, -1]],
    "target_image": "<base64 png>",
    "step_context": {
      "step_index": 3,
      "op": "draw_rectangle",
      "pending_numeric_text": null
    }
  }
}
```

Healthcheck body:

```json
{
  "input": {
    "healthcheck": true
  }
}
```
