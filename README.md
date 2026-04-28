# 👁️ Blinker – Eye Blink Detection & Reminder

A real-time eye blink tracker that uses your USB webcam to monitor blinking and reminds you to blink if you haven't in 10 seconds.

Built with **Python**, **OpenCV**, and **MediaPipe FaceLandmarker**.

## How It Works

1. **Face Mesh Detection** – MediaPipe's FaceLandmarker detects 468+ facial landmarks in real-time
2. **Eye Aspect Ratio (EAR)** – Calculates the ratio of vertical to horizontal eye distances; a low EAR means the eye is closed
3. **Blink Detection** – When EAR drops below the threshold for consecutive frames, a blink is registered
4. **Alert System** – If no blink is detected for 10 seconds, an audible alarm plays

## Setup

```bash
pip install -r requirements.txt
```

The `face_landmarker.task` model file is NOT included. If missing, download it:

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "face_landmarker.task"
```

## Usage

```bash
python blinker.py
```

## Controls

| Key     | Action                          |
|---------|---------------------------------|
| `Q`/`ESC` | Quit                          |
| `R`     | Reset blink counter             |
| `+`     | Increase EAR threshold (more sensitive) |
| `-`     | Decrease EAR threshold (less sensitive) |

## HUD Display

- **Top-left**: App title
- **Left panel**: Blink count, blinks/minute, current EAR value
- **Top-right**: Circular timer showing seconds since last blink (green → yellow → red)
- **Center banner**: Pulsing "BLINK NOW!" alert when 10s exceeded
- **Bottom**: Controls reference

## Configuration

Edit the constants at the top of `blinker.py`:

| Setting              | Default | Description                        |
|----------------------|---------|------------------------------------|
| `EAR_THRESHOLD`      | 0.21    | EAR below this = eyes closed       |
| `EAR_CONSEC_FRAMES`  | 2       | Frames below threshold for a blink |
| `ALERT_INTERVAL_SEC` | 10.0    | Seconds before alert               |
| `CAMERA_INDEX`       | 0       | Webcam device index                |

## Donations

If this tool helped you, consider a small donation to support my work:

PayPal: paypal@nexus-informatik.ch

## License

Copyright (c) 2026 Nexus Informatik Durrer. All rights reserved.
