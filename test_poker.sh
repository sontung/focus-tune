mkdir -p output/Cambridge_GreatCourt

# Per-cluster evaluation:
./test_ace.py datasets/Cambridge_GreatCourt ace_models/Cambridge_Ensemble/Cambridge_GreatCourt/0_4.pt --session 0_4
./test_ace.py datasets/Cambridge_GreatCourt ace_models/Cambridge_Ensemble/Cambridge_GreatCourt/1_4.pt --session 1_4
./test_ace.py datasets/Cambridge_GreatCourt ace_models/Cambridge_Ensemble/Cambridge_GreatCourt/2_4.pt --session 2_4
./test_ace.py datasets/Cambridge_GreatCourt ace_models/Cambridge_Ensemble/Cambridge_GreatCourt/3_4.pt --session 3_4

# Merging results and computing metrics.

# The merging script takes a --poses_suffix argument that's used to select only the
# poses generated for the requested number of clusters.
./merge_ensemble_results.py ace_models/Cambridge_Ensemble/Cambridge_GreatCourt output/Cambridge_GreatCourt/merged_poses_4.txt --poses_suffix "_4.txt"

# The output poses output by the previous script are then evaluated against the scene ground truth data.
./eval_poses.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/merged_poses_4.txt
