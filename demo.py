"""Convenience wrapper for generating the annotated demo video."""
from pathlib import Path
from main import process

if __name__ == "__main__":
    process(
        video="bowling_scoreboard.mp4",
        output_csv="output/scoreboard_results.csv",
        demo_video="output/demo_output.mp4",
        sample_every=15,
    )
    print("Demo created in output/demo_output.mp4")
