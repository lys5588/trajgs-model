# Consistency Analysis of the HumanML3D and Human3.6M 269-dim Unified Representations

## 1. Problem Statement

This analysis verifies whether, after converting HumanML3D and Human3.6M to the 269-dim unified representation, the physical meaning of each dimension is truly unified across the two data sources.

## 2. Recap of the 269-dim Layout

| index | dims | content | HumanML3D | Human3.6M |
|------|------|------|-----------|-----------|
| 0 | 1 | root_rot_velocity | actual value | 0 |
| 1-2 | 2 | root_linear_velocity | actual value | 0 |
| 3 | 1 | root_y | yes | yes |
| 4-66 | 63 | ric_data (21x3) | yes | yes |
| 67-192 | 126 | rot_data (21x6) | yes | yes |
| 193-258 | 66 | local_vel (22x3) | actual value | 0 |
| 259-262 | 4 | foot_contact | 0/1 | 0.5 |
| 263-268 | 6 | global_orient_6d | dynamic | **always identity rotation** |

## 3. Key Findings

### 3.1 global_orient_6d (last 6 dims) - NOT consistent

**Experimental data:**

- Human3.6M: 6D mean = `[1, 0, 0, 0, 1, 0]` (identity rotation), rotation angle = 0 degrees
- HumanML3D: integrating root_rot_vel gives a mean rotation angle of 49.98 degrees, max 90.37 degrees

**Root cause:**

1. **HumanML3D (`process_motion_generation`)**:
   - The whole sequence is rotated once using the **first frame's facing direction**
   - After IK, global_orient_6d reflects the **dynamic facing change relative to the first frame**
   - The first frame's global_orient is close to identity; subsequent frames record the facing evolution

2. **Human3.6M (`process_mesh_reconstruction`)**:
   - **Every frame is rotated independently** to face Z+
   - After IK, global_orient_6d is **always close to the identity rotation**
   - The original facing information is "normalized away"

**Conclusion:** the physical meaning of global_orient_6d is not unified. In HumanML3D it encodes dynamic facing; in Human3.6M it is always the identity rotation.

> Note: in the released training data, H36M is converted with `ROTATION_MODE=full` (see `scripts/h36m_smpl_to_269.sh`), which overrides rot_data (67:193) and global_orient_6d (263:269) with the real SMPL rotations. This partially mitigates the inconsistency described here, at the cost of a larger semantic gap from the HumanML3D IK semantics.

### 3.2 rot_data (dims 67-192, 21 joint rotations) - different reference frames

**Experimental data:**
```
joint |  Human3.6M  |  HumanML3D
--------------------------------
  0   |      51.69  |     120.56
  1   |      54.45  |     120.81
  9   |      15.00  |      98.40
 12   |      55.25  |     120.41
...
```

**Root cause:**

Although both compute rotations from joint positions via IK, the reference frames differ:

- HumanML3D: IK runs on the sequence normalized as a whole; all frames share one reference frame
- Human3.6M: each frame is normalized independently and has its own reference frame

Hence the same pose can yield different joint rotation values in the two modes.

**Conclusion:** the rot_data computation logic is identical, but the numerical distributions may differ because of the different reference frames.

### 3.3 ric_data (dims 4-66, 21 joint positions) - different reference frames but similar ranges

**Experimental data:**
```
axis |  Human3.6M (range)  |  HumanML3D (range)
---------------------------------------------
X    |  [-0.376, 0.347]    |  [-0.371, 0.491]
Y    |  [0.000, 1.512]     |  [0.000, 1.552]
Z    |  [-0.170, 0.299]    |  [-0.453, 0.555]
```

**Analysis:**

- The value ranges are similar, indicating the skeleton normalization is consistent
- The Y ranges are similar (0-1.5 m), indicating the axis conventions are consistent (head to +Y)
- But the reference frames still differ (HumanML3D uses the first frame's facing, Human3.6M is per-frame)

### 3.4 root_y (dim 3) - consistent

**Experimental data:**

- Human3.6M: mean = 0.8996, range = [0.8910, 0.9068]
- HumanML3D: mean = 0.9295, range = [0.8828, 0.9527]

**Conclusion:** both go through foot-on-floor processing; root_y has the same physical meaning.

### 3.5 Velocity features (dims 0-2, 193-258) - by design

- Human3.6M: all zeros (single frames carry no velocity information)
- HumanML3D: actual velocity values

**Conclusion:** intended by design; single-frame data has no notion of velocity.

### 3.6 Coordinate systems - unified

Both:

- head pointing to +Y (Human3.6M via `flip_coords`)
- skeleton normalized (`uniform_skeleton`)
- foot-on-floor applied
- facing +Z (but handled differently!)

## 4. Root Cause

The core issue is that `process_mesh_reconstruction` and `process_motion_generation` have different design goals:

| property | process_motion_generation | process_mesh_reconstruction |
|------|---------------------------|----------------------------|
| input | continuous motion sequence (T, 22, 3) | independent single frames (22, 3) |
| facing alignment | first frame aligned to Z+, whole sequence rotated | each frame aligned to Z+ independently |
| reference frame | unified (first frame's facing) | not unified (per frame) |
| global_orient | dynamic | always identity |
| purpose | motion generation | mesh reconstruction |

## 5. Impact Assessment

### 5.1 When used for motion generation tasks

**Issue:** Human3.6M's global_orient_6d being all-identity creates a distribution mismatch against HumanML3D data.

**Suggestions:**

- Option A: organize Human3.6M into sequences and process with `process_motion_generation`
- Option B: accept the current design and treat Human3.6M as "static poses facing Z+"

### 5.2 When used for mesh reconstruction tasks

**Small impact:** mesh reconstruction cares about single-frame poses and needs no temporal coherence.

- ric_data and rot_data have similar single-frame value ranges
- global_orient_6d in mesh reconstruction mode should be read as "the current frame's rotation relative to the canonical facing"

## 6. Code-level Evidence

The key code difference is in `unified_representation_269.py`:

```python
# process_motion_generation: the whole sequence uses the first frame's facing
root_quat_init = qbetween_np(forward_init, target)
root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init  # (T, 22, 4)
positions = qrot_np(root_quat_init, positions)  # one rotation for the whole sequence

# process_mesh_reconstruction: per-frame
for i in range(N):
    root_quat_init = qbetween_np(forward_init, np.array([[0, 0, 1]]))
    pos = qrot_np(root_quat_init_expanded, pos)  # independent rotation for this frame
```

## 7. Recommendations

### Short term
1. Document the design intent and differences of the two modes explicitly
2. Handle the two data sources separately during training

### Long term
1. For a truly unified representation, Human3.6M could:
   - be organized by video sequence
   - be processed with `process_motion_generation`
   - which would preserve the original facing changes

2. Or redesign the 269-dim representation:
   - define global_orient_6d as "absolute facing" instead of "rotation relative to the normalized facing"
   - but this would break compatibility with HumanML3D

## 8. Conclusion

**The physical meaning of the 269-dim representation is not fully unified across the two data sources.**

| dimension | consistency | note |
|------|--------|------|
| root_y | consistent | normalized root height in both |
| ric_data | similar ranges | different reference frames but comparable values |
| rot_data | similar ranges | different reference frames but identical computation |
| global_orient_6d | NOT consistent | dynamic in HumanML3D, always identity in Human3.6M (mitigated by ROTATION_MODE=full in the released data) |
| velocity features | by design | single frames have no velocity |
| foot_contact | by design | single frames use 0.5 (unknown) |

The main inconsistency is the meaning of **global_orient_6d**, which is determined by the different design goals of the two processing modes.
