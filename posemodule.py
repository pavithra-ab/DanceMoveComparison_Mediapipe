#import libraries
import cv2
import mediapipe as mp
import time

class poseDetector:
    def __init__(self, mode = False, upBody = False, smooth = True, detectionCon = 0.85, trackCon = 0.70):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose =self.mpPose.Pose(
            self.mode, self.upBody, self.smooth, False, self.detectionCon, self.trackCon
        )
    
    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
                )
        #print(self.results.pose_landmarks)
        return img
    
    def findPosition(self, img):
        self.lmlist = []

        if self.results.pose_landmarks:
            for id, lm in enumerate (self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(img.shape)
                print(id,lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])
            return self.lmlist
    

def main():
    video = cv2.VideoCapture(r'F:\DL Projects\Mediapipe Projects\Mediapipe_Project\dance_videos\right_dance.mp4')
    pTime = 0
    #desired_fps = 10
    #frame_delay = int(1000/desired_fps)
    detector = poseDetector()

    while True:
        succ, img = video.read()
        img = detector.findPose(img)
        lmlist = detector.findPosition(img)
        if lmlist != 0:
            print(lmlist[14])
            cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 15, (0,0,255), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img,
                    "FPS :" +str(int(fps)),
                    (70,50),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (0,255,0),
                    3)
        

        cv2.imshow("Image", img)
        #cv2.waitKey(frame_delay)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()        
        

