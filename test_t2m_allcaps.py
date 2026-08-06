# coding=utf-8
"""
T2M-only full-caption evaluation entrypoint.

This is a thin wrapper around `test_stage4_t2m_allcaps.py` that forces:
- `test.tasks = [t2m]`
- `test.t2m.use_all_captions = True`

So you can keep the same config/CLI shape as `test_stage4.py` and only
switch the script name.

Example:
torchrun --nproc_per_node=8 test_t2m_allcaps.py \
  config=configs/showo2_motion_test_stage4.yaml \
  test.tasks=[t2m] \
  test.save_vis.enabled=True
"""

import os

import test_stage4_t2m_allcaps as impl


_ORIGINAL_GET_CONFIG = impl.get_config


def _patched_get_config():
    config = _ORIGINAL_GET_CONFIG()

    requested_tasks = config.test.get("tasks", None)
    if isinstance(requested_tasks, str):
        requested_tasks = [requested_tasks]
    elif requested_tasks is None:
        requested_tasks = []
    else:
        requested_tasks = list(requested_tasks)

    if requested_tasks and requested_tasks != ["t2m"] and impl.is_main_process():
        print(
            f"[INFO] {os.path.basename(__file__)} only runs T2M. "
            f"Overriding test.tasks={requested_tasks} -> ['t2m']"
        )

    config.test.tasks = ["t2m"]

    if config.test.get("t2m", None) is None:
        config.test.t2m = {}
    config.test.t2m.use_all_captions = True

    if config.test.get("save_vis", None) is None:
        config.test.save_vis = {}
    config.test.save_vis.script_tag = os.path.splitext(os.path.basename(__file__))[0]
    if config.test.save_vis.get("save_t2m_sigma_smoothed_motion", None) is None:
        config.test.save_vis.save_t2m_sigma_smoothed_motion = True
    if config.test.save_vis.get("t2m_sigma_smoothed_task_name", None) is None:
        config.test.save_vis.t2m_sigma_smoothed_task_name = "t2m_sigma_smoothed"
    if config.test.save_vis.get("t2m_sigma_smoothed_sigma", None) is None:
        config.test.save_vis.t2m_sigma_smoothed_sigma = config.test.t2m.get(
            "latent_smooth_sigma", 1.5
        )

    if config.test.get("eval_title", None) is None:
        config.test.eval_title = "UniMotion Stage 4 T2M Full-Caption Evaluation"
    if config.test.get("eval_stage_label", None) is None:
        config.test.eval_stage_label = "STAGE 4 T2M ALLCAPS"
    if config.test.get("eval_result_prefix", None) is None:
        config.test.eval_result_prefix = "stage4_t2m_allcaps_eval"

    return config


impl.get_config = _patched_get_config


def main():
    impl.main()


if __name__ == "__main__":
    main()
