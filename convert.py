import os
import argparse

from converters.master import MasterConverter

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('-m', '--model', type=str, required=True, \
                        help='Path to keras model (.h5)')
    args.add_argument('--to-tflite', default=False, action='store_true')
    args.add_argument('--to-savedmodel', default=False, action='store_true')
    args.add_argument('--to-coreml', default=False, action='store_true')
    args.add_argument('--to-tftrt', default=False, action='store_true')
    args.add_argument('--all', default=False, action='store_true')
    args.add_argument('--fp16', default=False, action='store_true')

    args = args.parse_args()

    return args

def ensure_args(args):
    args_dict = vars(args)

    # check if model path valid
    model_path = args_dict.get('model')
    if not model_path.endswith('.h5'):
        raise ValueError('Unfortunately we only support conversion from keras model for now, so only .h5 file will be accepted')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'{model_path} not found on your system!')

    # check if at least one model selected
    passed = False
    for key in args_dict.keys():
        if key != 'model':
            passed = passed or args_dict.get(key)

            if passed:
                break
    
    if not passed:
        raise ValueError(f'Choose at least one target model type!')

if __name__ == '__main__':
    args = get_args()
    ensure_args(args)

    master_converter = MasterConverter(args)
    master_converter.process()
