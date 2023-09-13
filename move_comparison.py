import cv2, time
import posemodule as pm
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from ffpyplayer.player import MediaPlayer


def compare_positions(benchmark_video, user_video):

    benchmark = cv2.VideoCapture(benchmark_video)
    user = cv2.VideoCapture(user_video)
    #player = MediaPlayer(benchmark_video)

    fps = 0

    detector_1 = pm.poseDetector()
    detector_2 = pm.poseDetector()

    frame_counter = 0
    correct_frames = 0

    while benchmark.isOpened() or user.isOpened():
        try:
            succ_1, image_1 = user.read()
            
            #loop the video if it is ended, if the last frame is reached reset the capture and frame counters
            if frame_counter == user.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_counter = 0
                correct_frames = 0
                user.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            window_name = "User Video"
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name, 720, -10)

            image_1 = cv2.resize(image_1, (720,640))
            image_1 = detector_1.findPose(image_1)

            lmlist_user = detector_1.findPosition(image_1)
            del lmlist_user[1:11]

            succ_2, image_2 = benchmark.read()

            if frame_counter == benchmark.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_counter = 0
                correct_frames = 0
                benchmark.set(cv2.CAP_PROP_POS_FRAMES, 0)
                #player = MediaPlayer(benchmark_video)
                #continue

            # Play the audio using MediaPlayer
            #player.get_frame()
                

            image_2 = cv2.resize(image_2, (720,640))
            image_2 = detector_2.findPose(image_2)
            
            lmlist_benchmark = detector_2.findPosition(image_2)
            del lmlist_benchmark[1:11]
            
            frame_counter += 1

            if succ_2 or succ_1:
                error, _ = fastdtw(lmlist_user, lmlist_benchmark, dist = cosine)
                print(error)

                cv2.putText(
                    image_1,
                    "Error:{}%".format(str(round(100 * (float(error)), 2))),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

                if error < 0.4:
                    cv2.putText(
                        image_1,
                        "CORRECT STEPS",
                        (40, 600),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    correct_frames += 1
                
                else:
                    cv2.putText(
                        image_1,
                        "INCORRECT STEPS",
                        (40, 600),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                
                cv2.putText(
                    image_1,
                    "FPS: %f" % (1.0 / (time.time() - fps)),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2,
                )

                if frame_counter == 0:
                    frame_counter = user.get(cv2.CAP_PROP_FRAME_COUNT)

                cv2.putText(
                        image_1,
                        "Dance Steps Accurately Done: {}%".format(
                            str(round(100 * correct_frames / frame_counter, 2))
                        ),
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,255),
                        2,
                )

                
                cv2.imshow("Benchmark Video", image_2)
                cv2.imshow("User Video", image_1)

                fps = time.time()
                

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

            
                
        except:
            pass

    benchmark.release()
    user.release()
    cv2.destroyAllWindows()

    #player.close_player()



