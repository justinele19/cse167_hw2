# CSE 167 Homework 2 Submission

## My scene ("Make your own scene")

- **Scene:** `scenes/my_scene/ship_spin.json`
- **Rendering:** `outputs/my_scene.png`

A low-poly spaceship spins slowly in space, with a gray moon in the background. Sunlight comes from the upper right (a directional light), so the Phong shading from HW 2.7 (ambient + diffuse + specular) sweeps across the hull as it turns. The ship is tilted 15° toward the camera and makes a full 360° turn every 8 seconds (keyframe animation).

Reproduce with:
```bash
uv run python main.py 2_7 scenes/my_scene/ship_spin.json                                          # animated
uv run python main.py -output outputs/my_scene.png 2_7 -time 2.0 scenes/my_scene/ship_spin.json  # still image
```
The scene also renders with `2_5` and `2_6`, which use per-vertex colors and no lighting.

Files in `scenes/my_scene/`:
- `ship_spin.json`: the scene (camera, light, objects, materials, keyframes)
- `ship_metal.ply`, `ship_metalDark.ply`, `ship_dark.ply`, `ship_metalRed.ply`: the spaceship, split into one mesh per material color. All four use the same keyframes, so they move as one ship.
- `moon_sphere.json`: the moon (a copy of the provided `uv_sphere.json`)
- `CREDITS.txt`: model attribution

**Model credit:** the spaceship is `craft_speederD` from [Space Kit by Kenney](https://kenney.nl/assets/space-kit) (www.kenney.nl), licensed CC0. I converted it from OBJ to PLY with trimesh, split it into one PLY per material, centered and scaled it, and gave it flat-shaded normals. The colors (Kd) come from the model's original `.mtl` file.

## Bonus

None implemented.

## Required outputs

All graded renderings are in `outputs/` (`hw_2_2_*.png` through `hw_2_7_*.png`, plus `hw2_1_*.png`).

---

# balboa
UCSD CSE 167 codebase
https://cseweb.ucsd.edu/~tzli/cse167/

This is a revised version of balboa that builds on [SlangPy](https://github.com/shader-slang/slangpy). Currently only Homework 1 and 2 are completed. Future homeworks will be added soon.

# Setup
Clone the repository:
```bash
git clone https://github.com/BachiLi/balboa_slang_public.git
cd balboa_slang_public
```

We use [uv](https://docs.astral.sh/uv/) to manage Python packages and environments. Go to the linked website to install it if you haven't. Once installed, run
```bash
uv sync
```

Then try
```bash
uv run python main.py 1_1
```

You should see a window with white background.

# Platforms

Balboa depends on [SlangPy](https://github.com/shader-slang/slangpy) as a GPU programming framework, and should work on Windows, Linux, and Apple Silicon Macs. 

**Intel Macs:** SlangPy does not currently publish macOS x86_64 wheels, so setup may require building SlangPy from source. If you are using an Intel Mac and `uv sync` fails because no compatible SlangPy package is available, you may need to build SlangPy from source. Contact the staff/instructor if you run into issues.
