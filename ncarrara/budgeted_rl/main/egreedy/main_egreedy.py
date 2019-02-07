from ncarrara.budgeted_rl.main.utils import test_bftq, test_ftq, abstract_main
from ncarrara.budgeted_rl.main.egreedy import learn_ftq_egreedy, learn_bftq_egreedy
import sys

seeds = None
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    if len(sys.argv) >2:
        seed_start = int(sys.argv[2])
        number_seeds = int(sys.argv[3])
        seeds = range(seed_start, seed_start + number_seeds)
else:
    # config_file = "config/test.json"
    config_file = "config/test_highway.json"



def main():
    learn_ftq_egreedy.main()
    test_ftq.main()
    learn_bftq_egreedy.main()
    test_bftq.main()

override_param_grid = {
    'general.seed': seeds,
}

abstract_main.main(config_file, override_param_grid, main)

