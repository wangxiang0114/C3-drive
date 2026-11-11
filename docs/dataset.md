# C3-Drive dataset

## Directory layout

The default collector uses the CARLA 0.9.16 BasicAgent. Commands load NPZ episodes directly. Existing recordings can be imported after aligning their fields, timestamps, and splits with the conventions below.

```text
data/
  routes.json
  episodes/
    clean/
      train/0000.npz
      validation/0000.npz
      fit/0000.npz
      calibrate/0000.npz
      threshold/0000.npz
      test/0000.npz
    camera-drop/
      train/0000.npz
      fit/0000.npz
      calibrate/0000.npz
```

The `routes` command generates `routes.json` with the town and each route's start and end spawn indices, traffic seed, and split. Use the same manifest for clean and shifted observations. Store alternate recording configurations in separate directories.

## Collection and loading

```powershell
python -m c3drive collect --manifest data/routes.json --data data/episodes --domain clean --splits train validation fit calibrate threshold
python -m c3drive inspect-data --data data/episodes --domains clean --splits train validation fit calibrate threshold --output runs/dataset.json
python -m c3drive train-teacher --data data/episodes --output runs/teacher.pt
```

If you store episodes elsewhere, pass the same `--data` path to collection, inspection, training, offline evaluation, calibration, and adaptation. Run `inspect-data` after importing. It checks array shapes, data types, value ranges, and duplicate routes across splits, then reports episode and sample counts. Assign all recordings of a route to the same split, even if seeds or perturbations differ. Offline training can run on imported episodes independently. For closed-loop commands, the route manifest's spawn indices must match the loaded CARLA town.

## NPZ schema

Each file stores one episode. Arrays are synchronized along the first axis, where `N` is the number of samples. The sampling interval is 0.1 seconds.

| Field | Shape and type | Description |
| --- | --- | --- |
| `image`, `clean_image` | `[N, 9, 64, 112]`, `uint8` | Left, center, and right views concatenated in that order, with RGB channels per view. The fields store deployment observations and their synchronized clean counterparts. |
| `lidar` | `[N, 2, 48, 48]`, `uint8` | Bird's-eye-view point-count grid in vehicle coordinates, extending 24 m in each direction. The channels represent two height bands. |
| `state`, `clean_state` | `[N, 8]`, `float32` | Speed divided by 10, local `x, y` coordinates of two route points divided by 20, and left-turn, right-turn, and straight indicators. |
| `target` | `[N, 11]`, `float32` | Local `x, y` waypoint coordinates at 0.4, 0.8, 1.2, 1.6, and 2.0 seconds, divided by 10. The last value is the mean speed at those times divided by 10. |
| `severity` | `[N]`, `float32` | Severity based on collisions and solid-lane crossings, ranging from 0 to 1. |
| `critical` | `[N]`, `uint8` | Binary indicator for a collision or terminal route deviation within the target window. |
| `samples` | Integer scalar | The number of samples, `N`. |
| `metadata` | Unicode string scalar | JSON with at least `town`, `domain`, and `route`. The route object contains integer `start`, `end`, and `seed` fields. The collector also checks `requested_ticks` and `requested_traffic` when reusing recordings. |

In the local coordinate frame, positive `x` points forward and positive `y` points right. The loader forms a window from the current frame and the two preceding frames within each episode. At the beginning of an episode, it repeats the first frame as needed. Expert targets must come from the same trajectory. The collector drops observations near the end of each episode when their full future target window is unavailable. In the clean domain, the paired clean fields can match the corresponding deployment fields.

The code uses its own CARLA routes, data format, and scoring. To import data from a public benchmark, align sensor poses, sampling rate, labels, and units with this schema. Offline evaluation uses the labels defined above. Closed-loop scoring uses route progress and events recorded during policy execution.
