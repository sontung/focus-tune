import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from pykdtree.kdtree import KDTree
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from tqdm import tqdm

import colmap_read
from ace_util import get_pixel_grid, to_homogeneous, read_nvm_file
from ace_loss import ReproLoss
from ace_network import Regressor
from dataset import CamLocDataset
import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer

_logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class TrainerACE:
    def __init__(self, options):
        self.mse_errors = None
        self.options = options

        self.device = torch.device("cuda")

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
        # torch.backends.cuda.matmul.allow_tf32 = False

        # Setup randomness for reproducibility.
        self.base_seed = 2089
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        self.iteration = 0
        self.training_start = None
        self.num_data_loader_workers = 12

        # Create dataset.
        self.ds_name = str(self.options.scene).split("/")[-1]
        out_dir = Path(f"output/{self.ds_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        self.sfm_model_dir = None
        if "7scenes" in str(self.options.scene):
            self.ds_type = "7scenes"
            ds_name = str(self.options.scene).split("/")[-1].split("_")[-1]
            self.sfm_model_dir = f"../7scenes_reference_models/{ds_name}/sfm_gt"
            _logger.info(f"Reading SFM from {self.sfm_model_dir}")
            self.recon_images = colmap_read.read_images_binary(
                f"{self.sfm_model_dir}/images.bin"
            )
            self.recon_cameras = colmap_read.read_cameras_binary(
                f"{self.sfm_model_dir}/cameras.bin"
            )
            self.recon_points = colmap_read.read_points3D_binary(
                f"{self.sfm_model_dir}/points3D.bin"
            )
            self.image_name2id = {}
            for image_id, image in self.recon_images.items():
                self.image_name2id[image.name.replace("/", "-")] = image_id
            self.image_id2points = {}
            for img_id in self.recon_images:
                pid_arr = self.recon_images[img_id].point3D_ids
                pid_arr = pid_arr[pid_arr >= 0]
                xyz_arr = np.zeros((pid_arr.shape[0], 3))
                for idx, pid in enumerate(pid_arr):
                    xyz_arr[idx] = self.recon_points[pid].xyz
                self.image_id2points[img_id] = xyz_arr
            self.xyz_arr = np.zeros((4, 3))
        elif "Cambridge" in str(self.options.scene):
            self.ds_type = "Cambridge"
            _logger.info(
                f"Reading sfm from {self.options.scene / 'reconstruction.nvm'}"
            )
            self.xyz_arr, self.image2points, self.image2name = read_nvm_file(
                self.options.scene / "reconstruction.nvm"
            )
            self.name2id = {v: k for k, v in self.image2name.items()}

        a_dir = self.options.scene / "train"
        self.dataset = CamLocDataset(
            root_dir=a_dir,
            sfm_model_dir=self.sfm_model_dir,
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            use_half=self.options.use_half,
            image_height=self.options.image_resolution,
            augment=self.options.use_aug,
            aug_rotation=self.options.aug_rotation,
            aug_scale_max=self.options.aug_scale,
            aug_scale_min=1 / self.options.aug_scale,
            num_clusters=self.options.num_clusters,  # Optional clustering for Cambridge experiments.
            cluster_idx=self.options.cluster_idx,  # Optional clustering for Cambridge experiments.
        )

        _logger.info(
            f"Training with constraint masks radius={self.options.sampling_radius}"
        )
        _logger.info(f"Using {a_dir}")
        _logger.info(
            "Loaded training scan from: {} -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
                self.options.scene,
                len(self.dataset),
                self.dataset.mean_cam_center[0],
                self.dataset.mean_cam_center[1],
                self.dataset.mean_cam_center[2],
            )
        )

        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
        self.regressor = Regressor.create_from_encoder(
            encoder_state_dict,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous,
        )
        _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")

        self.regressor = self.regressor.to(self.device)
        self.regressor.train()

        # Setup optimization parameters.
        self.optimizer = optim.AdamW(
            self.regressor.parameters(), lr=self.options.learning_rate_min
        )

        # Setup learning rate scheduler.
        steps_per_epoch = self.options.training_buffer_size // self.options.batch_size
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.options.learning_rate_max,
            epochs=self.options.epochs,
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=False,
        )

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.options.use_half)

        # Generate grid of target reprojection pixel positions.
        pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE)
        self.pixel_grid_2HW = pixel_grid_2HW.to(self.device)

        # Compute total number of iterations.
        self.iterations = (
            self.options.epochs
            * self.options.training_buffer_size
            // self.options.batch_size
        )
        self.iterations_output = 100  # print loss every n iterations, and (optionally) write a visualisation frame

        # Setup reprojection loss function.
        self.repro_loss = ReproLoss(
            total_iterations=self.iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == "circle"),
        )

        # Will be filled at the beginning of the training process.
        self.training_buffer = None
        self.error_tracker = {}

    def train(self):
        """
        Main training method.

        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """

        # self.compute_map_cover(self.dataset2)

        creating_buffer_time = 0.0
        training_time = 0.0

        self.training_start = time.time()

        # Create training buffer.
        buffer_start_time = time.time()
        self.create_training_buffer()
        buffer_end_time = time.time()
        creating_buffer_time += buffer_end_time - buffer_start_time
        _logger.info(
            f"Filled training buffer in {buffer_end_time - buffer_start_time:.1f}s."
        )

        # ds_name = str(self.options.scene).split("/")[-1]
        # feasible_mask = np.load(f"output/{ds_name}/feasible_mask.npy").astype(bool)
        # all_pids = np.load(f"output/{ds_name}/all_pids.npy")
        # assert torch.sum(torch.abs(self.training_buffer["optimal_xyz_indices"]-all_pids)).item() == 0
        # for k in self.training_buffer:
        #     self.training_buffer[k] = self.training_buffer[k][feasible_mask]

        # Train the regression head.
        for self.epoch in range(self.options.epochs):
            epoch_start_time = time.time()
            self.run_epoch()
            training_time += time.time() - epoch_start_time

        end_time = time.time()
        _logger.info(
            f"Done without errors. "
            f"Creating buffer time: {creating_buffer_time:.1f} seconds. "
            f"Training time: {training_time:.1f} seconds. "
            f"Total time: {end_time - self.training_start:.1f} seconds."
        )
        self.save_model()

    def create_training_buffer(self):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Sampler.
        batch_sampler = sampler.BatchSampler(
            sampler.RandomSampler(self.dataset, generator=self.batch_generator),
            batch_size=1,
            drop_last=False,
        )

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # need to rescale all images in the batch to the same size).
        training_dataloader = DataLoader(
            dataset=self.dataset,
            sampler=batch_sampler,
            batch_size=None,
            worker_init_fn=seed_worker,
            generator=self.loader_generator,
            pin_memory=True,
            num_workers=self.num_data_loader_workers,
            persistent_workers=self.num_data_loader_workers > 0,
            timeout=60 if self.num_data_loader_workers > 0 else 0,
        )

        _logger.info("Starting creation of the training buffer.")

        # Create a training buffer that lives on the GPU.
        self.training_buffer = {
            "features": torch.empty(
                (self.options.training_buffer_size, self.regressor.feature_dim),
                dtype=(torch.float32, torch.float16)[self.options.use_half],
                device=self.device,
            ),
            "target_px": torch.empty(
                (self.options.training_buffer_size, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            "gt_poses_inv": torch.empty(
                (self.options.training_buffer_size, 3, 4),
                dtype=torch.float32,
                device=self.device,
            ),
            "intrinsics": torch.empty(
                (self.options.training_buffer_size, 3, 3),
                dtype=torch.float32,
                device=self.device,
            ),
            "intrinsics_inv": torch.empty(
                (self.options.training_buffer_size, 3, 3),
                dtype=torch.float32,
                device=self.device,
            ),
            "optimal_xyz_indices": torch.empty(
                (self.options.training_buffer_size,),
                dtype=torch.int64,
                device="cpu",
            ),
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()

        # The encoder is pretrained, so we don't compute any gradient.
        with torch.no_grad():
            # Iterate until the training buffer is full.
            buffer_idx = 0
            dataset_passes = 0

            while buffer_idx < self.options.training_buffer_size:
                dataset_passes += 1
                for (
                    image_B1HW,
                    image_ori,
                    image_mask_B1HW,
                    gt_pose_B44,
                    gt_pose_inv_B44,
                    gt_pose_sfm_inv_B44,
                    intrinsics_B33,
                    intrinsics_inv_B33,
                    _,
                    frame_path,
                    image_id_from_ds,
                    angle,
                    scale_factor,
                ) in training_dataloader:
                    # Copy to device.
                    image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                    intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                    intrinsics_inv_B33 = intrinsics_inv_B33.to(
                        self.device, non_blocking=True
                    )

                    # Compute image features.
                    with autocast(enabled=self.options.use_half):
                        features_BCHW = self.regressor.get_features(image_B1HW)

                    # Dimensions after the network's downsampling.
                    B, C, H, W = features_BCHW.shape

                    # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
                    image_mask_B1HW = TF.resize(
                        image_mask_B1HW,
                        [H, W],
                        interpolation=TF.InterpolationMode.NEAREST,
                    )

                    image_mask_B1HW = image_mask_B1HW.bool()

                    # If the current mask has no valid pixels, continue.
                    if image_mask_B1HW.sum() == 0:
                        continue

                    # Create a tensor with the pixel coordinates of every feature vector.
                    pixel_positions_B2HW = self.pixel_grid_2HW[
                        :, :H, :W
                    ].clone()  # It's 2xHxW (actual H and W) now.
                    pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
                    pixel_positions_B2HW = pixel_positions_B2HW.expand(
                        B, 2, H, W
                    )  # Bx2xHxW

                    # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
                    gt_pose_inv = gt_pose_inv_B44[:, :3]
                    gt_pose_inv = (
                        gt_pose_inv.unsqueeze(1)
                        .expand(B, H * W, 3, 4)
                        .reshape(-1, 3, 4)
                    )

                    # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
                    intrinsics = (
                        intrinsics_B33.unsqueeze(1)
                        .expand(B, H * W, 3, 3)
                        .reshape(-1, 3, 3)
                    )
                    intrinsics_inv = (
                        intrinsics_inv_B33.unsqueeze(1)
                        .expand(B, H * W, 3, 3)
                        .reshape(-1, 3, 3)
                    )

                    def normalize_shape(tensor_in):
                        """Bring tensor from shape BxCxHxW to NxC"""
                        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                    mask, uv_grid, uv_centroids = self.retrieve_gt_xyz(
                        image_ori,
                        image_id_from_ds.item(),
                        frame_path,
                        gt_pose_inv_B44,
                        intrinsics_B33,
                        H,
                        W,
                        angle,
                        scale_factor,
                        self.options.sampling_radius,
                    )

                    if torch.sum(mask) == 0:
                        continue

                    batch_data = {
                        "features": normalize_shape(features_BCHW),
                        "target_px": normalize_shape(pixel_positions_B2HW),
                        "gt_poses_inv": gt_pose_inv,
                        "intrinsics": intrinsics,
                        "intrinsics_inv": intrinsics_inv,
                    }

                    # Turn image mask into sampling weights (all equal).
                    image_mask_B1HW = image_mask_B1HW.float()
                    image_mask_N1 = normalize_shape(image_mask_B1HW)
                    image_mask_N1 = image_mask_N1 * mask.unsqueeze(1)

                    # Over-sample according to image mask.
                    features_to_select = self.options.samples_per_image * B
                    features_to_select = min(
                        features_to_select,
                        self.options.training_buffer_size - buffer_idx,
                    )

                    # Sample indices uniformly, with replacement.
                    sample_idxs = torch.multinomial(
                        image_mask_N1.view(-1),
                        features_to_select,
                        replacement=False
                        if features_to_select < torch.sum(image_mask_N1).item()
                        else True,
                        generator=self.sampling_generator,
                    )

                    # Select the data to put in the buffer.
                    for k in batch_data:
                        batch_data[k] = batch_data[k][sample_idxs]

                    # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                    buffer_offset = buffer_idx + features_to_select
                    for k in batch_data:
                        self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[
                            k
                        ]

                    buffer_idx = buffer_offset
                    if buffer_idx >= self.options.training_buffer_size:
                        break

        buffer_memory = sum(
            [v.element_size() * v.nelement() for k, v in self.training_buffer.items()]
        )
        buffer_memory /= 1024 * 1024 * 1024

        _logger.info(
            f"Created buffer of {buffer_memory:.2f}GB with {dataset_passes} passes over the training data."
        )
        self.regressor.train()

    def retrieve_gt_xyz(
        self,
        image_ori,
        image_id,
        frame_path,
        gt_pose_inv_B44,
        intrinsics_B33,
        H,
        W,
        angle,
        scale_factor,
        radius,
        vis=False,
    ):
        batch_idx = 0
        xyz = None
        if self.ds_type == "Cambridge":
            base_key = "/".join(
                frame_path[batch_idx].split("/")[-1].split(".png")[0].split("_")
            )
            image_key1 = f"{base_key}.jpg"
            image_id_from_map = None
            if image_key1 in self.name2id:
                image_id_from_map = self.name2id[image_key1]
            else:
                image_key2 = f"{base_key}.png"
                if image_key2 in self.name2id:
                    image_id_from_map = self.name2id[image_key2]

            pid_list = self.image2points[image_id_from_map]
            xyz = self.xyz_arr[pid_list]
        elif self.ds_type == "7scenes":
            base_key = frame_path[0].split("/")[-1]
            xyz = self.image_id2points[self.image_name2id[base_key]]

        xyzt = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        xyzt = torch.from_numpy(xyzt).permute([1, 0]).float().cuda()

        gt_inv_pose_34 = gt_pose_inv_B44[batch_idx, :3]
        cam_coords = torch.mm(gt_inv_pose_34, xyzt)
        uv = torch.mm(intrinsics_B33[batch_idx], cam_coords)
        uv[2].clamp_(min=0.1)  # avoid division by zero
        uv = uv[0:2] / uv[2]
        uv = uv.permute([1, 0]).cpu().numpy()

        if vis:
            image_ori2 = image_ori.numpy().astype(np.uint8)
            image_ori2 = image_ori2[0]
            for u, v in uv.astype(int):
                cv2.circle(image_ori2, (u, v), 5, (255, 0, 0), -1)
            cv2.imwrite(f"test2/im{image_id}.png", image_ori2)

        uv_grid = self.pixel_grid_2HW[:, :H, :W].clone()
        uv_grid_arr = uv_grid.view(2, -1).permute([1, 0]).cpu().numpy()

        b1, b2 = np.max(uv_grid_arr, 0)
        oob_mask1 = np.bitwise_and(0 <= uv[:, 0], uv[:, 0] < b1)
        oob_mask2 = np.bitwise_and(0 <= uv[:, 1], uv[:, 1] < b2)
        oob_mask = np.bitwise_and(oob_mask1, oob_mask2)

        tree = KDTree(uv[oob_mask])
        dis, ind = tree.query(uv_grid_arr)
        mask = dis < radius
        mask = mask.astype(int)
        mask = torch.from_numpy(mask).cuda()

        return mask, uv_grid_arr[dis < radius].astype(int), uv[oob_mask].astype(int)

    def run_epoch(self):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        # Shuffle indices.
        random_indices = torch.randperm(
            self.training_buffer["features"].shape[0], generator=self.training_generator
        )

        # Iterate with mini batches.
        buffer_size = self.training_buffer["features"].shape[0]
        for batch_start in range(0, buffer_size, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size

            # Drop last batch if not full.
            if batch_end > buffer_size:
                continue

            # Sample indices.
            random_batch_indices = random_indices[batch_start:batch_end]

            # Call the training step with the sampled features and relevant metadata.
            self.training_step(
                self.training_buffer["features"][random_batch_indices].contiguous(),
                self.training_buffer["target_px"][random_batch_indices].contiguous(),
                self.training_buffer["gt_poses_inv"][random_batch_indices].contiguous(),
                self.training_buffer["intrinsics"][random_batch_indices].contiguous(),
                self.training_buffer["intrinsics_inv"][
                    random_batch_indices
                ].contiguous(),
            )
            self.iteration += 1

    def training_step(
        self,
        features_bC,
        target_px_b2,
        gt_inv_poses_b34,
        Ks_b33,
        invKs_b33,
    ):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """
        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]

        # Reshape to a "fake" BCHW shape, since it's faster to run through the network compared to the original shape.
        features_bCHW = (
            features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)
        )
        with autocast(enabled=self.options.use_half):
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b31 = (
            pred_scene_coords_b3HW.permute(0, 2, 3, 1)
            .flatten(0, 2)
            .unsqueeze(-1)
            .float()
        )

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates.
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(
            reprojection_error_b2, dim=1, keepdim=True, p=1
        )

        #
        # Compute masks used to ignore invalid pixels.
        #
        # Predicted coordinates behind or close to camera plane.
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max

        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_b1 = invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1
        valid_mask_b1 = ~invalid_mask_b1

        # Reprojection error for all valid scene coordinates.
        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        # Compute the loss for valid predictions.
        loss_valid = self.repro_loss.compute(
            valid_reprojection_error_b1, self.iteration
        )

        # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(
            invKs_b33, pixel_grid_crop_b31
        )

        # Compute the distance to target camera coordinates.
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = (
            torch.abs(target_camera_coords_b31 - pred_cam_coords_b31)
            .masked_select(invalid_mask_b11)
            .sum()
        )

        # Final loss is the sum of all 2.
        loss = loss_valid + loss_invalid
        loss /= batch_size

        # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        old_optimizer_step = self.optimizer._step_count

        # Optimization steps.
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.iteration % self.iterations_output == 0:
            # Print status.
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)

            # mse_err = pred_scene_coords_b31.clone().detach().squeeze().cpu() - xyz_gt
            # mse_err = torch.mean(torch.abs(mse_err), 1)
            # mse_curr = torch.mean(mse_err)
            # self.mse_errors[xyz_gt_indices] = mse_err
            # for idx, pid in enumerate(xyz_gt_indices):
            #     self.error_tracker.setdefault(pid.item(), []).append(
            #         mse_err[idx].item()
            #     )
            # mse_err = torch.mean(self.mse_errors[self.mse_errors > -1])
            mse_err = -1
            mse_curr = -1
            _logger.info(
                f"Iteration: {self.iteration:6d} / Epoch {self.epoch:03d}|{self.options.epochs:03d}, "
                f"Loss: {loss:.1f}, Valid: {fraction_valid * 100:.1f}%, Time: {time_since_start:.2f}s"
            )

        # Only step if the optimizer stepped and if we're not
        # over-stepping the total_steps supported by the scheduler.
        if old_optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
            self.scheduler.step()

    def save_model(self):
        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats (~30MB).
        # torch.save(self.regressor.state_dict(), self.options.output_map_file)

        # This saves just the head weights as half-precision floating point numbers for a total of ~4MB, as mentioned
        # in the paper. The scene-agnostic encoder weights can then be loaded from the pretrained encoder file.
        head_state_dict = self.regressor.heads.state_dict()
        for k, v in head_state_dict.items():
            head_state_dict[k] = head_state_dict[k].half()
        torch.save(head_state_dict, self.options.output_map_file)
        _logger.info(f"Saved trained head weights to: {self.options.output_map_file}")
        _logger.info(f"Finished training for {str(self.options.scene)}")
        command_to_test = f"python test_ace.py {str(self.options.scene)} {str(self.options.output_map_file)}"
        _logger.info(f"Testing using {command_to_test}")
        os.system(
            f"python test_ace.py {str(self.options.scene)} {str(self.options.output_map_file)}"
        )

    def render_point_cloud(
        self,
        head_network_path="ace_models/wayspots/wayspots_squarebench.pt",
        dis_thresh=0.005,
    ):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Sampler.
        batch_sampler = sampler.BatchSampler(
            sampler.RandomSampler(self.dataset, generator=self.batch_generator),
            batch_size=1,
            drop_last=False,
        )

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # need to rescale all images in the batch to the same size).
        training_dataloader = DataLoader(
            dataset=self.dataset,
            sampler=batch_sampler,
            batch_size=None,
            worker_init_fn=seed_worker,
            generator=self.loader_generator,
            pin_memory=True,
            num_workers=self.num_data_loader_workers,
            persistent_workers=self.num_data_loader_workers > 0,
            timeout=60 if self.num_data_loader_workers > 0 else 0,
        )

        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
        _logger.info(f"Loaded encoder from: {self.options.encoder_path}")
        head_state_dict = torch.load(head_network_path, map_location="cpu")
        _logger.info(f"Loaded head weights from: {head_network_path}")

        # Create regressor.
        network = Regressor.create_from_split_state_dict(
            encoder_state_dict, head_state_dict
        )
        network = network.cuda()
        network.eval()

        arr_dir = Path(f"output/{self.ds_name}/map_gt.npy")
        if arr_dir.exists():
            all_points = np.load(str(arr_dir))
            rank_arr = np.load(f"output/{self.ds_name}/rank_gt.npy")
        else:
            all_points = None

            with torch.no_grad():
                for (
                    image_B1HW,
                    _,
                    gt_pose_B44,
                    gt_pose_inv_B44,
                    intrinsics_B33,
                    _,
                    _,
                    filenames,
                    _,
                    _,
                    _,
                ) in tqdm(training_dataloader, desc="Processing geometry"):
                    image_B1HW = image_B1HW.to("cuda", non_blocking=True)
                    scene_coordinates_B3HW = network(image_B1HW.float())
                    assert scene_coordinates_B3HW.size(0) == 1
                    scene_coords = (
                        scene_coordinates_B3HW.view(3, -1).permute([1, 0]).cpu().numpy()
                    )
                    if all_points is None:
                        all_points = scene_coords
                    else:
                        tree = KDTree(all_points)
                        distance, indices = tree.query(scene_coords)
                        mask = distance > dis_thresh
                        mask2 = distance <= dis_thresh
                        all_points[indices[mask2]] = 0.5 * (
                            all_points[indices[mask2]] + scene_coords[mask2]
                        )
                        scene_coords = scene_coords[mask]
                        all_points = np.vstack([all_points, scene_coords])

            tree = KDTree(all_points)
            rank_arr = np.zeros((all_points.shape[0],))
            with torch.no_grad():
                for (
                    image_B1HW,
                    _,
                    gt_pose_B44,
                    gt_pose_inv_B44,
                    intrinsics_B33,
                    _,
                    _,
                    filenames,
                    _,
                    _,
                    _,
                ) in tqdm(training_dataloader, desc="Computing ranks"):
                    image_B1HW = image_B1HW.to("cuda", non_blocking=True)
                    with autocast(enabled=True):
                        scene_coordinates_B3HW = network(image_B1HW)
                    assert scene_coordinates_B3HW.size(0) == 1
                    scene_coords = (
                        scene_coordinates_B3HW.view(3, -1).permute([1, 0]).cpu().numpy()
                    )
                    distance, indices = tree.query(scene_coords)
                    mask2 = distance < dis_thresh
                    rank_arr[indices[mask2]] += 1

            np.save(f"output/{self.ds_name}/map_gt.npy", all_points)
            np.save(f"output/{self.ds_name}/rank_gt.npy", rank_arr)

        xyz_arr = all_points[rank_arr > 1]
        tree = KDTree(xyz_arr)
        image_id2pid = {}
        with torch.no_grad():
            for (
                image_B1HW,
                _,
                gt_pose_B44,
                gt_pose_inv_B44,
                intrinsics_B33,
                _,
                _,
                filenames,
                image_id,
                _,
                _,
            ) in tqdm(training_dataloader, desc="Rendering points for each image"):
                image_B1HW = image_B1HW.to("cuda", non_blocking=True)
                with autocast(enabled=True):
                    scene_coordinates_B3HW = network(image_B1HW)
                assert scene_coordinates_B3HW.size(0) == 1
                scene_coords = (
                    scene_coordinates_B3HW.view(3, -1).permute([1, 0]).cpu().numpy()
                )
                distance, indices = tree.query(scene_coords)
                mask = distance < dis_thresh
                image_id2pid[image_id.item()] = indices[mask]
        return xyz_arr, image_id2pid
