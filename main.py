import cv2
import argparse
from src.detector import PlayerDetector
from src.tracker import BasicTracker
from src.feature_extractor import FeatureExtractor
from src.reidentifier import ReIdentifier


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Soccer Player Re-Identification")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="output/output_video.mp4", help="Path to save output video")
    parser.add_argument("--yolo-weights", type=str, default="models/yolo/best.pt", help="Path to YOLOv11 weights")
    return parser.parse_args()


def draw_boxes(frame, players):
    for player_id, bbox in players.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {player_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    detector = PlayerDetector(weights_path=args.yolo_weights)
    tracker = BasicTracker()
    extractor = FeatureExtractor()
    reidentifier = ReIdentifier(extractor)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_players(frame)
        tracked = tracker.update(detections)
        reidentified = reidentifier.reidentify(frame, tracked)

        output_frame = draw_boxes(frame, reidentified)
        out.write(output_frame)
        cv2.imshow("ReID Soccer", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
