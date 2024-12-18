import cv2
import numpy as np

class OFAYOLO:
    def __init__(self, dpu_runner, conf_threshold, nms_threshold):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.dpu_runner = dpu_runner
        
        self.num_classes = 80
        self.scale_factor = 0.00392156
        self.anchors = [
        [(10, 13), (16, 30), (33, 23)],
        [(30, 61), (62, 45), (59, 119)],
        [(116, 90), (156, 198), (373, 326)],
    ]
        self.input_tensors = dpu_runner.get_input_tensors()
        self.output_tensors = dpu_runner.get_output_tensors()
        print(self.input_tensors[0])
        print()
        [print(tensor) for tensor in self.output_tensors]
        self.input_shape = self.input_tensors[0].dims
        self.output_shapes = [
    (tensor.dims[1], tensor.dims[2]) for tensor in self.output_tensors
]
        self.class_labels = [
    "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train",
    "Truck", "Boat", "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter",
    "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear",
    "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase",
    "Frisbee", "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat",
    "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle",
    "Wine Glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple",
    "Sandwich", "Orange", "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut",
    "Cake", "Chair", "Couch", "Potted Plant", "Bed", "Dining Table", "Toilet",
    "TV", "Laptop", "Mouse", "Remote", "Keyboard", "Cell Phone", "Microwave",
    "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase",
    "Scissors", "Teddy Bear", "Hair Drier", "Toothbrush"
]

    def preprocess(self, i_frame):
        self.image_height, self.image_width = i_frame.shape[:2]        
        i_frame = cv2.resize(i_frame, (self.input_shape[2], self.input_shape[1]), interpolation=cv2.INTER_LINEAR)
        img_quantized = np.round(i_frame.astype(np.float32) * 2 ** self.input_tensors[0].get_attr("fix_point") * self.scale_factor).astype(np.int8)
        return img_quantized

    def run_dpu(self, img_quantized):
        fpga_input = [np.empty(self.input_shape, dtype=np.int8)]
        fpga_input[0][0, ...] = img_quantized
        fpga_output = [np.empty(tuple(tensor.dims[1:]), dtype=np.int8, order="C") for tensor in self.output_tensors]
        job_id = self.dpu_runner.execute_async(fpga_input, fpga_output)
        self.dpu_runner.wait(job_id)
        return fpga_output

    def postprocess(self, fpga_output):

        scaled_outputs = [
        (fpga_output[idx].astype(np.float32) / (2 ** tensor.get_attr("fix_point"))).reshape(tuple(tensor.dims))  
        for idx, tensor in enumerate(self.output_tensors)]

        reshaped_outputs = []
        for idx, scaled_output in enumerate(scaled_outputs):
            grid_height, grid_width = self.output_shapes[idx]
            expected_shape = (grid_height, grid_width, len(self.anchors[idx]), 5 + self.num_classes)
            reshaped_outputs.append(scaled_output.reshape(expected_shape))

        decoded_boxes = []
        for layer_idx, output in enumerate(reshaped_outputs):
            grid_height, grid_width, num_anchors, _ = output.shape
            anchors = np.array(self.anchors[layer_idx])  # Convert anchors to a NumPy array


            # Split the output into components
            tx, ty, tw, th, confidence = np.split(output[..., :5], 5, axis=-1)
            class_scores = output[..., 5:]
            

            # Apply sigmoid to confidence and class scores
            confidence = self.sigmoid(confidence).reshape(grid_height, grid_width, num_anchors)
            class_scores = self.sigmoid(class_scores)

            # Mask low-confidence boxes
            mask = confidence > self.conf_threshold  # Shape: (grid_height, grid_width, num_anchors)
            if not np.any(mask):
                continue  # Skip processing this layer if no detections meet the threshold

            

            # Prepare grid offsets for bx, by
            grid_x = np.arange(grid_width).reshape(1, grid_width, 1)  # Shape: (1, grid_width, 1)
            grid_y = np.arange(grid_height).reshape(grid_height, 1, 1)  # Shape: (grid_height, 1, 1)
            
            bx = (grid_x + self.sigmoid(tx).squeeze(-1)) / grid_width
            by = (grid_y + self.sigmoid(ty).squeeze(-1)) / grid_height
            bw = np.exp(tw).squeeze(-1) * anchors[:, 0] / self.input_shape[2]
            bh = np.exp(th).squeeze(-1) * anchors[:, 1] / self.input_shape[1]

            # Compute absolute coordinates
            x_min = (bx - bw / 2) * self.input_shape[2]
            y_min = (by - bh / 2) * self.input_shape[1]
            x_max = (bx + bw / 2) * self.input_shape[2]
            y_max = (by + bh / 2) * self.input_shape[1]
            

            # Apply mask to filter valid boxes
            x_min, y_min, x_max, y_max = [np.clip(arr[mask], 0, max_dim) for arr, max_dim in 
                              zip([x_min, y_min, x_max, y_max], [self.input_shape[2], self.input_shape[1], self.input_shape[2], self.input_shape[1]])]

            confidence = confidence[mask]
            class_scores = class_scores[mask]

            # Add decoded boxes to the list
            for i in range(len(x_min)):
                decoded_boxes.append({
                    "box": [x_min[i], y_min[i], x_max[i], y_max[i]],
                    "confidence": confidence[i],
                    "class_probs": class_scores[i]
                })
                
        return self.filter_and_nms_combined(decoded_boxes)
    
    def filter_and_nms(self, decoded_boxes):
    
        def calculate_iou(box1, box2):
            x1, y1, x2, y2 = box1
            x1_b, y1_b, x2_b, y2_b = box2
            inter_x1 = max(x1, x1_b)
            inter_y1 = max(y1, y1_b)
            inter_x2 = min(x2, x2_b)
            inter_y2 = min(y2, y2_b)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x2_b - x1_b) * (y2_b - y1_b)
            union_area = box1_area + box2_area - inter_area
            return inter_area / union_area if union_area > 0 else 0

        decoded_boxes.sort(key=lambda x: x["confidence"], reverse=True)
        nms_boxes = []
        while decoded_boxes:
            best_box = decoded_boxes.pop(0)
            nms_boxes.append(best_box)
            decoded_boxes = [
                box for box in decoded_boxes if calculate_iou(best_box["box"], box["box"]) < self.nms_threshold
            ]
        
        return nms_boxes
    
    def filter_and_nms_combined(self, decoded_boxes):
        """
        Combines metadata handling and NumPy optimization for non-max suppression.
        :param decoded_boxes: List of dictionaries with box metadata, where each box is 
                            represented as {"box": [x_min, y_min, x_max, y_max], 
                            "confidence": float, "class_probs": np.array}.
        :return: Filtered list of dictionaries with metadata for the selected boxes.
        """
        if not decoded_boxes:
            return []
        
        # Convert the decoded_boxes into NumPy arrays for faster processing
        boxes = np.array([box["box"] for box in decoded_boxes])  # Shape: (N, 4)
        confidences = np.array([box["confidence"] for box in decoded_boxes])  # Shape: (N,)
        
        # Sort by confidence in descending order
        idxs = np.argsort(-confidences)
        boxes = boxes[idxs]
        confidences = confidences[idxs]
        
        # Metadata reordering
        sorted_boxes = [decoded_boxes[i] for i in idxs]
        
        # Prepare to store selected boxes
        picked = []

        while len(boxes) > 0:
            # Select the box with the highest confidence
            picked.append(sorted_boxes[0])

            # Compute IoU of the top box with all other boxes
            x1 = np.maximum(boxes[0, 0], boxes[1:, 0])
            y1 = np.maximum(boxes[0, 1], boxes[1:, 1])
            x2 = np.minimum(boxes[0, 2], boxes[1:, 2])
            y2 = np.minimum(boxes[0, 3], boxes[1:, 3])

            inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            box_area = (boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3] - boxes[0, 1])
            areas = (boxes[1:, 2] - boxes[1:, 0]) * (boxes[1:, 3] - boxes[1:, 1])
            union_area = box_area + areas - inter_area

            iou = inter_area / np.maximum(union_area, 1e-6)

            # Remove boxes with IoU > threshold
            keep = np.where(iou < self.nms_threshold)[0]

            # Update boxes and metadata
            boxes = boxes[keep + 1]
            sorted_boxes = [sorted_boxes[i + 1] for i in keep]

        return picked


    def draw_detections(self, frame, detections):

        scale_x = self.input_shape[2] / self.image_width
        scale_y = self.input_shape[1] / self.image_height
        
        for detection in detections:
            x_min, y_min, x_max, y_max = detection["box"]
            x_min = max(0, int(x_min / scale_x))
            y_min = max(0, int(y_min / scale_y))
            x_max = min(self.image_width, int(x_max / scale_x))
            y_max = min(self.image_height, int(y_max / scale_y))
            class_probs = detection["class_probs"]
            class_id = np.argmax(class_probs)
            class_score = detection["class_probs"][class_id]
            color = self.get_color(class_id)
            confidence = detection["confidence"]
            label = f"{self.class_labels[class_id]} ({class_score*confidence:.2f})"
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
            cv2.putText(frame, label, (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def get_color(self, label):
        # Use a predefined color map for consistency
        colors = [
            (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
            (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
            (188, 189, 34), (23, 190, 207)
        ]
        return colors[label % len(colors)]
    
    def get_color_0(self, label):
        return (int(label * 2 % 256), int(255 - label * 2 % 256), int((label + 50) % 256))   