from random_walk_single_metric import workflow
from random_walk_single_metric.config import RandomWalkSingleMetricConfig

if __name__ == '__main__':
    workflow(RandomWalkSingleMetricConfig().parse_args())
