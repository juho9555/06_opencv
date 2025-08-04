import cv2
import numpy as np

# 초기설정
img1 = None  # ROI로 선택한 참조 이미지
win_name = 'Camera Matching'
MIN_MATCH = 10

# ORB 특징점 추출기
detector = cv2.ORB_create(nfeatures=1500)

# FLANN 매칭기 설정 (LSH 기반 ORB에 적합)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# 카메라 캡처
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

match_status = ""  # 현재 매칭 상태 메시지

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if img1 is None:
        res = frame.copy()
        cv2.putText(res, "Press SPACE to select label ROI", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        img2 = frame
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)

        # 특징점 추출 확인
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            match_status = "No descriptors — Cannot match"
            res = frame.copy()
        else:
            matches = matcher.knnMatch(desc1, desc2, k=2)

            # 좋은 매칭점 필터링
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            print('good matches: %d / %d' % (len(good_matches), len(matches)))
            matchesMask = None
            res = frame.copy()

            if len(good_matches) > MIN_MATCH:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if mtrx is not None:
                    accuracy = float(mask.sum()) / mask.size
                    print("accuracy: %d/%d (%.2f%%)" % (mask.sum(), mask.size, accuracy * 100))

                    if mask.sum() > MIN_MATCH:
                        matchesMask = [int(x) for x in mask.ravel()]
                        h, w = img1.shape[:2]
                        pts = np.float32([[[0, 0]], [[0, h-1]], [[w-1, h-1]], [[w-1, 0]]])
                        dst = cv2.perspectiveTransform(pts, mtrx)
                        img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                        match_status = "Label MATCHED!"
                    else:
                        match_status = "Different labels"
                else:
                    match_status = "Homography failed"
            else:
                match_status = "Not enough good matches"

            # 결과 그리기
            res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                  matchColor=(0, 255, 0),
                                  matchesMask=matchesMask,
                                  flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        # 결과 텍스트 출력
        cv2.putText(res, match_status, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),  2)

    # 결과 출력
    cv2.imshow(win_name, res)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC 종료
        break
    elif key == ord(' '):  # 스페이스바 → ROI 선택
        print("[INFO] Select ROI for label.")
        x, y, w, h = cv2.selectROI(win_name, frame, False)
        if w and h:
            img1 = frame[y:y+h, x:x+w]
            print(f"[INFO] ROI selected: ({x}, {y}, {w}, {h})")
        else:
            print("[WARN] ROI not selected.")

cap.release()
cv2.destroyAllWindows()
