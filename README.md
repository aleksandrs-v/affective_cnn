# Affective Tactile Gesture Recognition Pipeline

An end-to-end MATLAB software pipeline for developing, training, and evaluating compact 1D convolutional neural networks (CNNs) for affective tactile gesture recognition from multichannel capacitive and inertial sensor data.

## Overview

The pipeline covers the complete workflow from Bluetooth-streamed data acquisition through interactive quality control and preprocessing to large-scale automated architecture search and live sensor-driven inference. It was developed for use with sensorized soft companion devices but is adaptable to any Bluetooth-streamed multichannel time-series sensor system.

For a detailed description of the software, see the companion publication:

Vališevskis, A. (2025). MATLAB Software Pipeline for Affective Tactile Gesture Recognition Using Compact 1D Convolutional Neural Networks. *SoftwareX*. \[DOI TBD\]

The pipeline was used to design and validate the model reported in:

Vališevskis, A. et al. (2025). Design and Validation of a Lightweight 1D CNN for Affective Touch Classification in Soft Plush Companions. arXiv preprint arXiv:\[ID TBD\].

## Pipeline Stages

| Stage | Script | Description |
| :---- | :---- | :---- |
| 1 | `Processing/PlushSamples.pde` | Data acquisition GUI (Processing/Java). Connects to the sensorized device via Bluetooth serial, displays live sensor channels, and records timestamped gesture samples. |
| 2 | `MATLAB/a_move_labelled_data.m` | Reorganizes raw session files from a person-centric to a class-centric directory layout and anonymizes filenames. |
| 3 | `MATLAB/b_training_data_analysis.m` | Quality control: computes per-class recording length statistics and visualizes outliers. |
| 4 | `MATLAB/c_training_data_truncation.m` | Interactive GUI for rapid truncation of reaction-time latency from recorded signals. Click-on-waveform \+ Enter workflow processes hundreds of samples in minutes. |
| 5 | `MATLAB/d_data_preparation.m` | Feature reduction (13 to 11 channels), train/validation/test splitting, and windowing. Two modes: sliding-window segmentation (default) or uniform-length loop-based padding. |
| 6 | `MATLAB Experiment Manager/*.mlx` \+ `MATLAB/e_train.m` | Automated hyperparameter sweep across CNN architectures via Experiment Manager, or manual single-architecture training via e\_train.m. Includes LSTM and BiLSTM baselines. |
| 7 | `MATLAB/f_realtime_classifier.m` | Real-time PC-based classifier GUI. Streams live sensor data via Bluetooth, displays per-class prediction likelihoods with EMA smoothing. |

## Requirements

- **MATLAB R2025a** or later  
- **Deep Learning Toolbox** (required for Stages 6 and 7\)  
- **Processing 4** (required only for Stage 1, the data acquisition tool)  
- Operating system: Windows, macOS, or Linux

No additional MATLAB toolboxes are required.

## Installation

1. Clone this repository:  
     
   git clone https://github.com/aleksandrs-v/affective\_cnn.git  
     
2. Open MATLAB and navigate to the `MATLAB/` directory.  
3. Run scripts sequentially from `a_move_labelled_data.m` through `f_realtime_classifier.m`, adjusting configuration parameters at the top of each script to match your data and hardware.

For Stage 1 (data acquisition), open `Processing/PlushSamples.pde` in the Processing IDE and upload to run.

## Configuration

Key parameters that must be adjusted for your specific setup:

| Parameter | Script | Default | Description |
| :---- | :---- | :---- | :---- |
| Source/target directories | `a_move_labelled_data.m` | \-- | Paths to raw and reorganized training data |
| `GESTURES` array | `a_move_labelled_data.m` | 17 classes | List of gesture class names |
| `thresholdRange` | `b_training_data_analysis.m` | 0.4 | Outlier detection threshold (fraction of class mean) |
| `winLen` | `d_data_preparation.m` | 250 | Sliding window length (samples) |
| `hop` | `d_data_preparation.m` | 50 | Sliding window hop stride (samples) |
| COM port | `f_realtime_classifier.m` | COM7 | Bluetooth serial port for the sensorized device |
| `alpha` | `f_realtime_classifier.m` | 0.2 | EMA smoothing coefficient |

## Symbolic CNN Architecture DSL

Network topologies are specified as compact strings parsed at runtime by Experiment Manager. For example:

14d1\_39d2\_41d4

defines a three-layer dilated 1D CNN with kernel widths of 14, 39, and 41 samples and dilation factors of 1, 2, and 4 respectively. Filter counts are controlled independently through a global filter-scaling multiplier.

## License

This project is licensed under the MIT License. See [LICENSE](http://LICENSE) for details.

## Citation

If you use this software in your research, please cite:

@article{valisevskis2025pipeline,

  author  \= {Vali{\\v{s}}evskis, Aleksandrs},

  title   \= {{MATLAB} Software Pipeline for Affective Tactile Gesture Recognition Using Compact {1D} Convolutional Neural Networks},

  journal \= {SoftwareX},

  year    \= {2025},

  note    \= {DOI TBD}

}

## Contact

Aleksandrs Vališevskis \-- [Aleksandrs.Valisevskis@rtu.lv](mailto:Aleksandrs.Valisevskis@rtu.lv)

Institute of Architecture and Design, Riga Technical University, Riga, Latvia  
