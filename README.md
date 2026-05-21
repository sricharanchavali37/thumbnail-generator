\# Automatic Thumbnail Generator



Production grade backend that automatically generates

high quality thumbnails from any uploaded video.



\---



\## What It Does



User uploads a video → system analyses every frame using

computer vision → returns 5 best thumbnail candidates →

user picks one → saved to their profile history forever.



Supports any video format. Any file size. Any duration.

Built to handle 1000+ concurrent users.



\---



\## Tech Stack



| Layer | Technology |

|---|---|

| Backend API | FastAPI (Python) |

| Job Queue | Celery + Redis |

| Video Processing | FFmpeg + OpenCV |

| Database | MongoDB |

| Storage | AWS S3 |

| Deployment | Docker + AWS EC2 |



\---



\## How It Works



1\. Video uploaded → streamed directly to AWS S3

2\. Celery task fired per video into Redis queue

3\. Worker streams frames from S3 via FFmpeg

4\. OpenCV scores every frame on 5 criteria:

&#x20;  blur, brightness, faces, motion, transitions

5\. Content type auto-detected: vlog / sports / cinematic / general

6\. Scoring weights adjusted per content type

7\. Top 5 frames selected spread across full video

8\. Thumbnails resized to 1280x720 uploaded to S3

9\. Frontend polls status every 3 seconds until done



\---



\## Key Technical Decisions



\*\*Streaming not downloading\*\*

FFmpeg reads S3 URL directly. Zero disk usage.

Works for 100MB or 100GB video with same memory.



\*\*Parallel processing\*\*

One Celery task per video. All run simultaneously.

One failure never affects other videos.



\*\*Content aware scoring\*\*

Algorithm detects video type and adjusts weights.

Vlog prioritises faces. Sports prioritises motion.

Cinematic prioritises brightness.



\*\*Crash recovery\*\*

task\_acks\_late ensures no job is lost if worker dies.

Job goes back to queue automatically.



\---



\## API Endpoints

POST   /videos/upload                     Upload videos

GET    /videos/status?job\_ids=id1,id2     Poll job status

PATCH  /videos/{job\_id}/select-thumbnail  Save chosen thumbnail

GET    /videos/history                    Get upload history

POST   /videos/{job\_id}/regenerate        Re-generate thumbnails

GET    /videos/{job\_id}                   Single video detail

GET    /health                            Server health check



## Project Structure

backend/

├── main.py              Entry point, startup, health check

├── config.py            Environment variables from .env

├── database/

│   └── mongodb.py       Connection, indexes, pooling

├── models/

│   └── video.py         Video schema, enums, response models

├── routers/

│   └── videos.py        All 6 API endpoints

├── services/

│   └── s3.py            All S3 operations

├── workers/

│   ├── celery\_app.py    Celery + Redis configuration

│   └── video\_processor.py  Full FFmpeg + OpenCV pipeline

├── middleware/

│   └── auth.py          JWT user\_id extraction

└── utils/

└── ffmpeg\_helper.py FFmpeg streaming utilities

---



\## Local Setup



```bash

pip install -r requirements.txt

cp .env.example .env

docker run -d -p 6379:6379 --name redis-local redis:alpine

uvicorn backend.main:app --reload

```



Visit http://127.0.0.1:8000/docs to see all endpoints.



