import usleep
import utime
import os


# GLOBALS
PREDICTIONS_DIR = "predictions"
CHANNELS = """"O1-M2"==EEG "O2-M1"==EEG "E1-M1"==EOG "E2-M1"==EOG"""
GROUPING = "EOG EEG"
NUM_GPUS = 0
LOG_DIR = "logs"
MODEL = "u-sleep:1.0"

# Helpers
set_log_path = lambda x: os.path.join(LOG_DIR, x.replace(".npy",".log"))
set_predictions = lambda x: os.path.join(PREDICTIONS_DIR,x) if PREDICTIONS_DIR not in x else x

def predict_one(input_file, output_file, data_per_prediction):

    cmd = (
            f"ut predict_one"
            f" -f {input_file}"
            f" -o {set_predictions(output_file)}"
            f" --channels {CHANNELS}"
            f" --auto_channel_grouping {GROUPING}"
            f" --model {MODEL}"
            f" --logging_out_path {set_log_path(output_file)}"
            f" --num_gpus {NUM_GPUS}"
            f" --data_per_prediction {data_per_prediction}"
            f" --no_argmax"
            f" --overwrite"
        )

    print(cmd)
    out = os.system(cmd)
    
    return "call failed..." if out > 0 else "call successful"


