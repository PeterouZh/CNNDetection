import os
import sys
import torch
import torch.nn
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from networks.resnet import resnet50


def build_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-f','--file', default='examples_realfakedir')
  parser.add_argument('-m','--model_path', type=str, default='weights/blur_jpg_prob0.5.pth')
  parser.add_argument('-c','--crop', type=int, default=None, help='by default, do not crop. specify crop size')
  parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')
  return parser

# opt = parser.parse_args()


def main(opt, myargs):
  model = resnet50(num_classes=1)
  state_dict = torch.load(opt.model_path, map_location='cpu')
  model.load_state_dict(state_dict['model'])
  if(not opt.use_cpu):
    model.cuda()
  model.eval()

  # Transform
  trans_init = []
  if(opt.crop is not None):
    trans_init = [transforms.CenterCrop(opt.crop),]
    print('Cropping to [%i]'%opt.crop)
  else:
    print('Not cropping')
  trans = transforms.Compose(trans_init + [
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  img = trans(Image.open(opt.file).convert('RGB'))

  with torch.no_grad():
      in_tens = img.unsqueeze(0)
      if(not opt.use_cpu):
        in_tens = in_tens.cuda()
      prob = model(in_tens).sigmoid().item()

  print('probability of being synthetic: {:.2f}%'.format(prob * 100))
  pass



def run(argv_str=None):
  from template_lib.utils.config import parse_args_and_setup_myargs, config2args
  from template_lib.utils.modelarts_utils import prepare_dataset
  run_script = os.path.relpath(__file__, os.getcwd())
  args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
  myargs.args = args1
  myargs.config = getattr(myargs.config, args1.command)

  if hasattr(myargs.config, 'datasets'):
    prepare_dataset(myargs.config.datasets, cfg=myargs.config)

  parser = build_parser()
  opt = parser.parse_args([])

  opt = config2args(myargs.config.args, opt)

  main(opt, myargs)
  pass

if __name__ == '__main__':
  run()
  from template_lib.examples import test_bash
  test_bash.TestingUnit().test_resnet(gpu=os.environ['CUDA_VISIBLE_DEVICES'])