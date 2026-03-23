# VideoCAD

This website is inspired by the repositories for the [Nerfies website](https://nerfies.github.io).

**VideoCAD** is a large-scale video dataset for learning user interface (UI) interactions and 3D reasoning from CAD software.  
It contains over 41,000 CAD modeling videos with synchronized user actions, enabling multimodal learning for CAD automation, 3D understanding, and human–AI collaboration.

If you find **VideoCAD** useful for your work, please cite:

```bibtex
@inproceedings{manvideocad,
  title={VideoCAD: A Dataset and Model for Learning Long-Horizon 3D CAD UI Interactions from Video},
  author={Man, Brandon and Nehme, Ghadi and Alam, Md Ferdous and Ahmed, Faez},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track}
}
```

---

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
  <img alt="Creative Commons License" style="border-width:0"
       src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" />
</a><br />
This website is licensed under a
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
Creative Commons Attribution-ShareAlike 4.0 International License</a>.

## Autonomous Onshape CLI

This repo now also contains a separate Python package under `src/videocad_onshape` for a constrained autonomous Onshape runtime built around VideoCADFormer.

CLI commands:

```bash
videocad-onshape run --prompt "Create a centered rectangle 10 by 20 and extrude it 5."
videocad-onshape dry-run --prompt "Create a centered rectangle 10 by 20 and extrude it 5."
videocad-onshape calibrate
videocad-onshape healthcheck
```

Configuration is loaded from a TOML file plus environment overrides. Start from [videocad_onshape.example.toml](/Users/mtkachenko/Documents/Borzo/Videocadf/videocad.github.io/videocad_onshape.example.toml) and set:

- browser profile / blank Part Studio URL
- planner API key and model
- Runpod endpoint id and API key
- fixed window geometry
- safety thresholds
- output directory

You can also place secrets and machine-local paths in a repo-local `.env` file. Start from [.env.example](/Users/mtkachenko/Documents/Borzo/Videocadf/videocad.github.io/.env.example). The loader reads `.env` from the config directory or repo root before applying environment overrides.
For Chrome, `VIDEOCAD_ONSHAPE_BROWSER_PROFILE_PATH` may point either to the Chrome user-data root or directly to a `Default` / `Profile N` directory. If needed, set `VIDEOCAD_ONSHAPE_BROWSER_PROFILE_NAME` explicitly as well.

## Runpod Inference

Local execution remains on macOS, but VideoCADFormer inference is now expected to run remotely on Runpod. The repo includes a deployable worker under [runpod_worker/README.md](/Users/mtkachenko/Documents/Borzo/Videocadf/videocad.github.io/runpod_worker/README.md) that loads the released checkpoint and serves `runsync` requests from the local CLI.
