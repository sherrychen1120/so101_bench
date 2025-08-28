from so101_bench.task_progress_labeler import TaskProgressLabeler

def test_task_progress_labeler():
    episode_dir = "/home/melon/sherry/so101_bench/tests/assets/task_progress_labeler"
    video_paths = [
        episode_dir + "/video_cam_front.mp4",
        episode_dir + "/video_cam_top.mp4",
    ]
    progress_stage_labels = ["stage_1", "stage_2", "stage_3"]
    fps = 30.0
    horizon_s = 10.0
    player = TaskProgressLabeler(video_paths, progress_stage_labels, fps, horizon_s, "test episode")
    progress_stages = player.play()

    print(progress_stages)

if __name__ == "__main__":
    test_task_progress_labeler()
