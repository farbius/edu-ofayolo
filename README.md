## **OFA YOLO Implementation for ZynqMP UltraScale+ with Vitis AI**

This repository contains implementations of the OFA YOLO object detector for ZynqMP UltraScale+ devices using Vitis AI. It leverages the OFA YOLO model from the Vitis AI model zoo, optimized for the B1152 architecture of the DPU.
### **Prerequisites**

- [ZynqMP UltraScale+ 2CG device with B1152 DPU architecture](https://www.fpga-radar.com/vivado-hw-dpu)
- [Linux for Zynq Ultrascale+ with Vitis AI 3.0](https://www.fpga-radar.com/petalinux-vitis-ai)
- [Compiled OFA YOLO models for DPU](https://github.com/farbius/edu-vitis-ai.git)
- USB camera Logitech C720

The `test_slow_ofayolo.py` script provides a baseline implementation of the OFA YOLO detector. It processes the input sequentially without optimizations. 

| Argument          | Default Value                  | Description                                         |
|-------------------|--------------------------------|-----------------------------------------------------|
| `-m, --xmodel`    | `ofa_yolo_pt.xmodel` | Path to the OFA YOLO model file.                   |
| `-c, --conf_threshold` | `0.35`                    | Confidence threshold for detections.               |
| `-n, --nms_threshold` | `0.10`                    | Non-Maximum Suppression (NMS) threshold.           |
| `-i, --input`     | `0`                            | Input source identifier (camera, video, or image). |
Compiled models for dpu engine:

`ofa_yolo_pt.xmodel`
`ofa_yolo_pruned_0_30_pt.xmodel`
`ofa_yolo_pruned_0_50_pt.xmodel`

Run the script with default arguments:
```bash
python test_slow_ofayolo.py
python test_slow_ofayolo.py -m ofa_yolo_pt.xmodel -i example_video.avi
python test_slow_ofayolo.py -i example_image.jpg
python test_slow_ofayolo.py -i 0
python test_slow_ofayolo.py -c 0.5 -n 0.2
```
