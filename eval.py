import os
import csv
import torch

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *


def main(opt, myargs):
  # Running tests
  # opt = TestOptions().parse(print_options=False)
  model_name = os.path.basename(opt.model_path).replace('.pth', '')
  rows = [["{} model testing on...".format(model_name)],
          ['testset', 'accuracy', 'avg precision']]

  print("{} model testing on...".format(model_name))
  for v_id, val_dict in enumerate(opt.vals):
    val = list(val_dict.keys())[0]
    print(f'Model: {val}')
    opt.dataroot = '{}/{}'.format(opt.datadir, val)
    opt.classes = os.listdir(opt.dataroot) if val_dict[val] else ['']
    opt.no_resize = True  # testing without resizing by default

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, _, _, _, _ = validate(model, opt, stdout=myargs.stdout)
    rows.append([val, acc, ap])
    print("({}) acc: {}; ap: {}".format(val, acc, ap))

  csv_name = myargs.args.outdir + '/{}.csv'.format(model_name)
  with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)


def run(argv_str=None):
  from template_lib.utils.config import parse_args_and_setup_myargs, config2args
  from template_lib.utils.modelarts_utils import prepare_dataset
  run_script = os.path.relpath(__file__, os.getcwd())
  args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
  myargs.args = args1
  myargs.config = getattr(myargs.config, args1.command)

  if hasattr(myargs.config, 'datasets'):
    prepare_dataset(myargs.config.datasets, cfg=myargs.config)

  opt = TestOptions().parse(print_options=False)

  opt = config2args(myargs.config.args, opt)

  main(opt, myargs)
  pass


if __name__ == '__main__':
  run()
  from template_lib.examples import test_bash
  test_bash.TestingUnit().test_resnet(gpu=os.environ['CUDA_VISIBLE_DEVICES'])
