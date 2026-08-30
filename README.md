# Scoreboard Data Extraction from Video

Computer Vision solution for extracting bowling scoreboard information from the supplied `bowling_scoreboard.mp4` video.

## Objective

Detect the scoreboard when it is visible, identify the active player, and extract the active player's cumulative **TTL (total)** from the video frames.

## Approach

1. **Video processing:** OpenCV reads the input video frame by frame.
2. **Scoreboard presence detection:** Template matching on the fixed `TTL` header region prevents animation/non-scoreboard frames from being processed.
3. **Active row detection:** HSV color segmentation detects the yellow-highlighted active player row.
4. **Player identification:** The yellow player-name header is matched against reference templates. Tesseract OCR is used as a fallback when the template confidence is low.
5. **TTL extraction:** The white TTL digits are isolated from the fixed right-hand ROI. Shape/template matching verifies the OCR-style digit reading against the supplied video's bowling font.
6. **Temporal filtering:** The system records only changes in the detected active player/TTL, reducing duplicate results.
7. **Output:** Timestamped results are saved to CSV and an annotated demo video is generated.

## Technologies

- Python 3.10+
- OpenCV
- NumPy
- Tesseract OCR / pytesseract
- CSV

## Project structure

```text
scoreboard_cv_project/
├── main.py
├── demo.py
├── requirements.txt
├── README.md
├── .gitignore
├── bowling_scoreboard.mp4   # supplied input video; optional in GitHub
└── output/
    ├── scoreboard_results.csv
    └── demo_output.mp4       # generated demo
```

## Installation

Install Python packages:

```bash
pip install -r requirements.txt
```

Install **Tesseract OCR** separately and make sure `tesseract` is available in PATH.

## Run

Place `bowling_scoreboard.mp4` in the project root and run:

```bash
python main.py --video bowling_scoreboard.mp4 --output output/scoreboard_results.csv --demo output/demo_output.mp4
```

Or use:

```bash
python demo.py
```

## Sample extracted output

The solution was tested on the supplied video. Representative detected updates are:

| Time (s) | Active row | Player | TTL |
|---:|:---:|---|---:|
| 0.0 | T | TARUN | 33 |
| 7.5 | T | TARUN | 36 |
| 26.5 | J | JAGDISH | 31 |
| 36.0 | V | VISHAL | 28 |
| 52.5 | V | VISHAL | 37 |

The full machine-readable result is stored in `output/scoreboard_results.csv`.

## Notes

The scoreboard layout and reference templates are tuned to the provided assessment video. This is intentional because the assessment supplies a fixed-format input video. The pipeline can be generalized to other scoreboard layouts by updating the ROIs and reference templates.
