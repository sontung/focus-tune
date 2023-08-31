DS="Cambridge_ShopFacade"

mkdir -p output/$DS

# Head training:
./train_ace.py datasets/"$DS" output/"$DS"/0_4.pt --num_clusters 4 --cluster_idx 0 --constraint_mask 1
./train_ace.py datasets/"$DS" output/"$DS"/1_4.pt --num_clusters 4 --cluster_idx 1 --constraint_mask 1
./train_ace.py datasets/"$DS" output/"$DS"/2_4.pt --num_clusters 4 --cluster_idx 2 --constraint_mask 1
./train_ace.py datasets/"$DS" output/"$DS"/3_4.pt --num_clusters 4 --cluster_idx 3 --constraint_mask 1

# Per-cluster evaluation:
./test_ace.py datasets/"$DS" output/"$DS"/0_4.pt --session 0_4
./test_ace.py datasets/"$DS" output/"$DS"/1_4.pt --session 1_4
./test_ace.py datasets/"$DS" output/"$DS"/2_4.pt --session 2_4
./test_ace.py datasets/"$DS" output/"$DS"/3_4.pt --session 3_4

# Merging results and computing metrics.

# The merging script takes a --poses_suffix argument that's used to select only the 
# poses generated for the requested number of clusters. 
./merge_ensemble_results.py output/"$DS" output/"$DS"/merged_poses_4.txt --poses_suffix "_4.txt"

# The output poses output by the previous script are then evaluated against the scene ground truth data.
./eval_poses.py datasets/"$DS" output/"$DS"/merged_poses_4.txt
