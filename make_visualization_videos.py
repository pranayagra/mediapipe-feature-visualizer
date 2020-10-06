import os
import glob
import argparse

from utility_scripts.make_mediapipe_video import make_mediapipe_video
from utility_scripts.make_features_table import make_features_table
from utility_scripts.json_data import load_json

def make_visualization_videos(input_frames_directory, input_mediapipe_directory, input_feature_config_filepath, users, output_visualization_directory, feature_type, table_video, visualization_types, frame_rate):

	features_config = load_json(input_feature_config_filepath)
	features = features_config[feature_type]

	if not users:
		features_filepaths = glob.glob(os.path.join(input_mediapipe_directory, '**.data'), recursive = True)
	else:
		features_filepaths = []
		for user in users:
			features_filepaths.extend(glob.glob(os.path.join(input_mediapipe_directory, f'*{user}**.data'), recursive = True))

	if table_video:
		print("Making feature table for each trial")
		make_features_table(input_mediapipe_directory, users, output_visualization_directory, 'trials')

	 
	for features_filepath in features_filepaths:
		filename = features_filepath.split('/')[-1]
		session, phrase, trial, _ = filename.split('.')
		
		frames_directory = os.path.join(input_frames_directory, session, phrase, trial, '*.png')
		save_directory = os.path.join(output_visualization_directory, 'videos', session, phrase, trial)
		table_filepath = os.path.join(output_visualization_directory, 'tables/trials', session, phrase, trial, '{}.{}.{}.png'.format(session, phrase, trial))

		if not os.path.exists(save_directory):
			os.makedirs(save_directory)
			make_mediapipe_video(frames_directory, features_filepath, save_directory, features, table_video, table_filepath, visualization_types, frame_rate)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_frames_directory', type = str, default = 'input/DATA/Frames')
	parser.add_argument('--input_mediapipe_directory', type = str, default = 'input/DATA/Mediapipe')
	parser.add_argument('--input_feature_config_filepath', type = str, default = 'input/configs/features.json')
	parser.add_argument('--users', nargs='*', default = None)
	parser.add_argument('--output_visualization_directory', type = str, default = 'output/visualization')
	parser.add_argument('--feature_type', type = str, default = 'visualization_features')
	parser.add_argument('--table_video', action = 'store_true', default = True)
	parser.add_argument('--visualization_types', nargs='*', default = [['mediapipe'], ['interpolate']])
	parser.add_argument('--frame_rate', type = int, default = 5)
	args = parser.parse_args()

	make_visualization_videos(args.input_frames_directory, args.input_mediapipe_directory, args.input_feature_config_filepath, args.users, args.output_visualization_directory, args.feature_type, args.table_video, args.visualization_types, args.frame_rate)
