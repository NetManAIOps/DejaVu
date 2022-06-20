from random_walk_failure_instance import workflow
from random_walk_failure_instance.config import RandomWalkFailureInstanceConfig

if __name__ == '__main__':
    workflow(RandomWalkFailureInstanceConfig().parse_args())
