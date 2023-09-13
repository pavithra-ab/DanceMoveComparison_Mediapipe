from move_comparison import compare_positions


benchmark_video = r"F:\DL Projects\Mediapipe Projects\Mediapipe_Project\dance\benchmark_dance.mp4"
#user_video = r'F:\DL Projects\Mediapipe Projects\Mediapipe_Project\dance\benchmark_dance.mp4'
user_video = r'F:\DL Projects\Mediapipe Projects\Mediapipe_Project\dance\right_dance.mp4'
#benchmark_video = r'F:\DL Projects\Mediapipe Projects\Mediapipe_Project\dance_videos\benchmark_dance.mp4'
#user_video = r'F:\DL Projects\Mediapipe Projects\Mediapipe_Project\dance_videos\right_dance.mp4'
compare_positions(benchmark_video, user_video)