# LLMOps_basicProject
Basic project understanding the concepts of AL Evaluations (Evals)

To trigger evals on CircleCI app use the below code syntax:

from utils import trigger_commit_evals
trigger_commit_evals(git_repo, git_branch, cci_api_key)
