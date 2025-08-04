# OpenCV

## 01. 이미지 매칭 (평균 해시 매칭, 템플릿 매칭)
- 이미지 간 유사도 측정 기법
- 평균 해시(average hash)를 통한 빠른 이미지 비교
- 템플릿 매칭은 이미지 내 특정 패턴 위치 찾기
- **코드 참고:** `01.avg_hash.py`, `03.template_matching.py`

## 02. 이미지의 특징점과 특징점 검출기 (Keypoints detector)
- 이미지 내 특징적인 점(코너, 엣지 등) 검출
- 검출된 특징점은 객체 인식 및 추적의 기초
- 다양한 검출기 제공 (FAST, Harris 등)
- **코드 참고:** `05.corner_goodFeature.py`, `06.kpt_gfft.py`
<img width="533" height="382" alt="image" src="https://github.com/user-attachments/assets/563f633c-64c0-460b-8b92-66f328e6c878" />



## 03. 특징 디스크립터 검출기 (SIFT, SURF, ORB)
- 특징점 검출과 기술을 위한 알고리즘
- SIFT, SURF는 강력하지만 속도 제약 존재
- ORB는 빨라서 실시간 처리에 적합
- **코드 참고:** `07.kpt_fast.py`, `08.desc_surf.py`  
<img width="533" height="382" alt="image" src="https://github.com/user-attachments/assets/a7945a49-43b1-4e3a-9e7b-e37ff5acab21" />


## 04. 특징 매칭 (Feature Matching)
- 서로 다른 두 이미지 간 특징점과 디스크립터 매칭
- SIFT, SURF, ORB 같은 특징점 검출기 사용
- 특징점과 디스크립터 비교를 통해 이미지 내 같은 객체 검출
- **코드 참고:** `07.kpt_fast.py`, `08.desc_sift.py`

## 05. 올바른 매칭점 찾기 (Good Match Points)
- 특징점 매칭에서 잘못된 매칭점 제거 필요
- Lowe’s 비율 테스트 활용 (두 번째 가까운 매칭점과 거리 비교)
- 매칭 품질 향상을 위한 필터링 과정
- **코드 참고:** `09.desc_surf.py`, `10.camera_matching.py`  
<img width="850" height="512" alt="image" src="https://github.com/user-attachments/assets/6a2f71ee-de42-4c7d-83ac-8fe498658905" />

## 06. 객체 추적을 위한 Tracking API (Tracking API)
- 객체 추적을 위한 OpenCV의 Tracking API 소개
- 다양한 추적 알고리즘 및 적용 방법
- 알고리즘 원리 이해 없이도 쉽게 추적 기능 사용 가능
- **코드 참고:** `08.match_track` 폴더 내 Tracking API 관련 예제

## 07. 배경 제거 (Background Subtraction)
- 동영상에서 지속적으로 움직이는 객체 추적 시 배경 제거 필요성
- 배경 모델링을 통해 움직이는 객체만 분리
- 다양한 배경 제거 알고리즘 존재 (예: MOG, KNN)
- **코드 참고:** `13.track_bgsb_mog.py`
<img width="642" height="512" alt="image" src="https://github.com/user-attachments/assets/a6782d44-797c-4e32-8ddd-a8ba7d36c3bb" />


## 08. 광학 흐름 (Optical Flow)
- 영상 내 물체의 움직임 패턴 분석 기법
- 프레임 간 픽셀 이동 방향과 거리 측정
- 객체의 움직임 감지 및 경로 추적에 활용
- 대표 알고리즘: Lucas-Kanade Optical Flow
- **코드 참고:** `14.track_opticalLK.py`
<img width="770" height="608" alt="image" src="https://github.com/user-attachments/assets/12bdd08a-d4ad-490f-9293-b21de2a99001" />
