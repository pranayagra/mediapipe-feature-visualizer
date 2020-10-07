# mediapipe-feature-visualizer
A utility tool to visualize features such as bounding boxes and landmarks in order to analyze object recognition performance and construct informed decisions.

## Getting started

Populate `input/DATA/Mediapipe` with MediaPipe `.data` files. The filename must follow the following convention: `<session>.<phrase>.<trial>.data`.

Populate `input/DATA/Frames/<session>/<phrase>/<trial>` with frames of the raw video that correspond to the Mediapipe data file `<session>.<phrase>.<trial>.data`. The frames in sequential order must follow the following filename convention: `frame_{i:03d}.png` where `{i:03d}` is the 3-digit number starting from `000`. For example, the first few frames will be named `frame_000.png, frame_001.png, frame_002.png, ...`.

## Running the Pipeline

Open a terminal and execute (most basic version of command): `python3 make_visualization_videos.py`.

This will create visualization videos and frequency tables in the `output/visualization` directory for all `.data` files.

## Optional Edits

* `make_visualization_videos.py` `users`: Specify a list of users (keywords) to run the visualization on a refined list of `.data` files.  
* `make_visualization_videos.py` `table_video`: Flag to specify whether or not to generate table images and append onto the visualization video.
* `make_visualization_videos.py` `table_video`: The frame rate of the visualization video.
* `make_visualization_videos.py` `visualization_types`: The models to use in the visualization video(s).  
* `make_visualization_videos.py` `other arguments`: Other arguments are unlikely to change if the setup above is followed. 
* `input/configs/features.json`: The features to display on the visualization.
* `utility_scripts/make_features_table.py`: This script is ran with `mode = trials` if `table_video` is set. However, other modes can also be set from the script directly to display tables at various levels of abstraction.
* `utility_scripts/model_feature_data_extraction/custom_feature_data.py`: A mostly empty script that is integrated into the visualization pipeline (remember to add 'custom' to `visualization_types`). This is made for your convenience if you want to experiment with different data post-processing models. 

## Future Work

* Release a script that can convert a dataset of videos into frames in the appropriate format. 
* Extend visualizer to allow additional features.
* Improve upon the interpolation feature data extraction model.





