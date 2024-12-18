import argparse
import cv2
import numpy as np
import vart
import xir
import time
from ofa_yolo import OFAYOLO


def get_child_subgraph_dpu(graph):
        root_subgraph = graph.get_root_subgraph()
        child_subgraphs = root_subgraph.toposort_child_subgraph()
        return [cs for cs in child_subgraphs if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"]


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--xmodel", required=False,	default="ofa_yolo_pruned_0_50_pt.xmodel", 
                help = "ofa yolo xmodel (default = ofa_yolo_pruned_0_50_pt.xmodel)")
ap.add_argument("-c", "--conf_threshold", required=False, type=float, default=0.35,	
                help = "ofa yolo detector confidence threshold (default = 0.35)")
ap.add_argument("-n", "--nms_threshold", required=False, type=float, default=0.10, 
                help = "ofa yolo detector NMS threshold (default = 0.10)")
ap.add_argument("-i", "--input", required=False, type=str, default="0",	
                help = "input source identifier (default = 0)")
args = vars(ap.parse_args())

def main():

    xmodel = args["xmodel"]
    print('<< ofa yolo detector - xmodel = ',xmodel)
    conf_threshold = args["conf_threshold"]
    print('<< ofa yolo detector - confidence threshold = ',conf_threshold)
    nms_threshold = args["nms_threshold"]
    print('<< ofa yolo detector - NMS threshold = ',nms_threshold)
    inputId = args["input"]
    print('<< input source identifier = ',inputId)

    # Timing accumulators for each stage
    fps = 0
    capture_time_total = 0
    preprocess_time_total = 0
    inference_time_total = 0
    postprocess_time_total = 0
    draw_time_total = 0
    frame_count = 0

    graph = xir.Graph.deserialize(xmodel)
    subgraphs = get_child_subgraph_dpu(graph)
    assert len(subgraphs) == 1, "Expected exactly one DPU subgraph."
    dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")

    detector = OFAYOLO(dpu_runner, conf_threshold, nms_threshold)

    # Determine the type of input source
    if inputId.isdigit():  # Camera input
        print(f"[INFO] Opening camera with ID {inputId}")
        cap = cv2.VideoCapture(int(inputId), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open camera {inputId}")
            exit()

    elif inputId.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video file
        print(f"[INFO] Opening video file {inputId}")
        cap = cv2.VideoCapture(inputId)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video file {inputId}")
            exit()

    elif inputId.lower().endswith(('.jpg', '.jpeg', '.png')):  # Image file
        print(f"[INFO] Loading image file {inputId}")
        frame = cv2.imread(inputId)
        if frame is None:
            print(f"[ERROR] Failed to load image file {inputId}")
            exit()
        # Process the image
        # Measure preprocessing time
        preprocess_start = time.time()
        img_quantized = detector.preprocess(frame)
        preprocess_end = time.time()
        preprocess_time_total += (preprocess_end - preprocess_start)

        # # Measure inference time
        inference_start = time.time()
        fpga_output = detector.run_dpu(img_quantized)
        inference_end = time.time()
        inference_time_total += (inference_end - inference_start)

        # Measure postprocessing time
        postprocess_start = time.time()
        detections = detector.postprocess(fpga_output)
        postprocess_end = time.time()
        postprocess_time_total += (postprocess_end - postprocess_start)

        # Measure drawing/rendering time
        draw_start = time.time()
        result_frame = detector.draw_detections(frame, detections)
        draw_end = time.time()
        draw_time_total += (draw_end - draw_start)

        cv2.imshow("ofa-yolo detector", result_frame)
        key = cv2.waitKey(0)
        del dpu_runner
        cv2.destroyAllWindows()
        print(f"{'Preprocessing:':<20} {preprocess_time_total * 1000:.2f} ms")
        print(f"{'Run DPU:':<20} {inference_time_total * 1000:.2f} ms")
        print(f"{'Postprocessing:':<20} {postprocess_time_total * 1000:.2f} ms")
        print(f"{'Rendering/Draw:':<20} {draw_time_total * 1000:.2f} ms")

        return  # Exit after processing a single image

    else:
        print(f"[ERROR] Unsupported input source: {inputId}")
        exit()
    
    start_time = time.time()

    try:
        while True:
            # Measure frame capture time
            capture_start = time.time()
            ret, frame = cap.read()
            # print(frame.shape[:2])
            capture_end = time.time()
            if not ret:
                break
            capture_time_total += (capture_end - capture_start)

            # Measure preprocessing time
            preprocess_start = time.time()
            img_quantized = detector.preprocess(frame)
            preprocess_end = time.time()
            preprocess_time_total += (preprocess_end - preprocess_start)

            # # Measure inference time
            inference_start = time.time()
            fpga_output = detector.run_dpu(img_quantized)
            inference_end = time.time()
            inference_time_total += (inference_end - inference_start)

            # Measure postprocessing time
            postprocess_start = time.time()
            detections = detector.postprocess(fpga_output)
            postprocess_end = time.time()
            postprocess_time_total += (postprocess_end - postprocess_start)

            # Measure drawing/rendering time
            draw_start = time.time()
            result_frame = detector.draw_detections(frame, detections)
            draw_end = time.time()
            draw_time_total += (draw_end - draw_start)

            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = frame_count / elapsed_time

            # Display FPS on the frame
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 180), 2)

            # Display the result
            cv2.imshow("ofa-yolo detector", result_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Press 'ESC' to exit
                break

    
    finally:
        # Stop the postprocessing thread
        del dpu_runner
        cap.release()
        cv2.destroyAllWindows()

        # Print average timing for each stage
        print("\nAverage Timings per Frame (in ms):")
        print(f"{'Frame Capture:':<20} {capture_time_total / frame_count * 1000:.2f} ms")
        print(f"{'Preprocessing:':<20} {preprocess_time_total / frame_count * 1000:.2f} ms")
        print(f"{'Run DPU:':<20} {inference_time_total / frame_count * 1000:.2f} ms")
        print(f"{'Postprocessing:':<20} {postprocess_time_total / frame_count * 1000:.2f} ms")
        print(f"{'Rendering/Draw:':<20} {draw_time_total / frame_count * 1000:.2f} ms")
        print(f"{'Overall FPS:':<20} {fps:.2f}")


if __name__ == "__main__":
    main()
