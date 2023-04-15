from training.profiler_trainer import NetworkxStatsTrainer, StatsTrainer
from training.trainer import *
from training.utils import *



def get_trainer(args):
    if args.do_profile:
        if args.model == 'networkx':
            return NetworkxStatsTrainer.from_args(args)
        else:
            return StatsTrainer.from_args(args)
    elif args.do_test:
        return TestReinforceTrainer.from_args(args)
    elif args.train_mode == 'reinforce':
        return BaselineReinforceTrainer.from_args(args)
    else:
        raise NotImplementedError()
