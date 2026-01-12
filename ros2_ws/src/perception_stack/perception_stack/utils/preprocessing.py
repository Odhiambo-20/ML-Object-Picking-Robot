#!/usr/bin/env python3
"""
Image preprocessing utilities for object detection pipeline.
Production-grade preprocessing with advanced augmentation, normalization,
and quality assurance for industrial vision systems.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
from enum import Enum
import random
import time
from scipy import ndimage
from skimage import exposure, filters, restoration
from skimage.morphology import disk, opening, closing
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import imgaug.augmenters as iaa

# Configure logging
logger = logging.getLogger(__name__)

class ImageFormat(Enum):
    """Supported image formats."""
    BGR = "bgr"
    RGB = "rgb"
    GRAY = "gray"
    YUV = "yuv"
    HSV = "hsv"
    LAB = "lab"

class NormalizationMethod(Enum):
    """Normalization methods."""
    STANDARD = "standard"      # (x - mean) / std
    MINMAX = "minmax"          # (x - min) / (max - min)
    ZERO_MEAN = "zero_mean"    # x - mean
    UNIT_STD = "unit_std"      # x / std
    PER_CHANNEL = "per_channel" # Normalize each channel independently
    IMAGENET = "imagenet"      # ImageNet statistics

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    # Input parameters
    input_height: int = 640
    input_width: int = 640
    input_channels: int = 3
    input_format: ImageFormat = ImageFormat.BGR
    
    # Normalization
    normalization_method: NormalizationMethod = NormalizationMethod.IMAGENET
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet mean
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)   # ImageNet std
    
    # Augmentation
    enable_augmentation: bool = True
    augmentation_probability: float = 0.5
    augmentation_strength: float = 0.5
    
    # Quality control
    min_image_size: Tuple[int, int] = (32, 32)
    max_image_size: Tuple[int, int] = (4096, 4096)
    quality_threshold: float = 0.5
    
    # Performance
    use_gpu: bool = False
    batch_size: int = 32
    num_workers: int = 4
    
    # Advanced features
    enable_histogram_equalization: bool = False
    enable_denoising: bool = False
    enable_sharpening: bool = False
    enable_contrast_enhancement: bool = False
    enable_color_correction: bool = False
    
    # Debug
    debug_mode: bool = False
    save_preprocessed: bool = False
    save_path: str = "/tmp/preprocessed"

class ImagePreprocessor:
    """
    Advanced image preprocessor for object detection.
    Supports multiple augmentation strategies, normalization methods,
    and quality control for production environments.
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.setup_logging()
        self.setup_augmentation_pipeline()
        self.setup_normalization()
        
        # Performance tracking
        self.processing_times = []
        self.frame_count = 0
        
        # Quality metrics
        self.quality_scores = []
        
        logger.info("ImagePreprocessor initialized with config: %s", self.config)
    
    def setup_logging(self):
        """Configure logging for preprocessor."""
        log_level = logging.DEBUG if self.config.debug_mode else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def setup_augmentation_pipeline(self):
        """Setup augmentation pipeline based on configuration."""
        self.augmentation_pipeline = None
        
        if not self.config.enable_augmentation:
            return
        
        try:
            # Define augmentation transforms
            transforms = []
            
            # Geometric transformations
            if self.config.augmentation_strength > 0.3:
                transforms.extend([
                    A.RandomRotate90(p=0.5 * self.config.augmentation_strength),
                    A.Flip(p=0.5 * self.config.augmentation_strength),
                    A.Transpose(p=0.3 * self.config.augmentation_strength),
                ])
            
            # Spatial transformations
            transforms.extend([
                A.ShiftScaleRotate(
                    shift_limit=0.1 * self.config.augmentation_strength,
                    scale_limit=0.1 * self.config.augmentation_strength,
                    rotate_limit=30 * self.config.augmentation_strength,
                    p=0.5 * self.config.augmentation_probability
                ),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50 * self.config.augmentation_strength,
                    alpha_affine=50 * self.config.augmentation_strength,
                    p=0.3 * self.config.augmentation_probability
                ),
                A.GridDistortion(p=0.3 * self.config.augmentation_probability),
            ])
            
            # Color transformations
            transforms.extend([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2 * self.config.augmentation_strength,
                    contrast_limit=0.2 * self.config.augmentation_strength,
                    p=0.5 * self.config.augmentation_probability
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20 * self.config.augmentation_strength,
                    sat_shift_limit=30 * self.config.augmentation_strength,
                    val_shift_limit=20 * self.config.augmentation_strength,
                    p=0.5 * self.config.augmentation_probability
                ),
                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=0.3 * self.config.augmentation_probability
                ),
                A.CLAHE(
                    clip_limit=4.0 * self.config.augmentation_strength,
                    tile_grid_size=(8, 8),
                    p=0.3 * self.config.augmentation_probability
                ),
            ])
            
            # Noise and blur
            transforms.extend([
                A.GaussNoise(
                    var_limit=(10.0, 50.0) * self.config.augmentation_strength,
                    p=0.3 * self.config.augmentation_probability
                ),
                A.GaussianBlur(
                    blur_limit=(3, 7),
                    p=0.3 * self.config.augmentation_probability
                ),
                A.MedianBlur(
                    blur_limit=7,
                    p=0.3 * self.config.augmentation_probability
                ),
                A.MotionBlur(
                    blur_limit=7,
                    p=0.3 * self.config.augmentation_probability
                ),
            ])
            
            # Weather and lighting effects (for robustness)
            transforms.extend([
                A.RandomFog(
                    fog_coef_lower=0.1,
                    fog_coef_upper=0.3,
                    alpha_coef=0.08,
                    p=0.2 * self.config.augmentation_probability
                ),
                A.RandomShadow(
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    shadow_roi=(0, 0.5, 1, 1),
                    p=0.2 * self.config.augmentation_probability
                ),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 1),
                    angle_lower=0,
                    angle_upper=1,
                    num_flare_circles_lower=6,
                    num_flare_circles_upper=10,
                    src_radius=400,
                    src_color=(255, 255, 255),
                    p=0.1 * self.config.augmentation_probability
                ),
            ])
            
            # Create pipeline
            self.augmentation_pipeline = A.Compose(
                transforms,
                p=self.config.augmentation_probability
            )
            
            logger.info("Augmentation pipeline created with %d transforms", len(transforms))
            
        except Exception as e:
            logger.error("Failed to create augmentation pipeline: %s", str(e))
            self.augmentation_pipeline = None
    
    def setup_normalization(self):
        """Setup normalization parameters."""
        if self.config.normalization_method == NormalizationMethod.IMAGENET:
            self.mean = np.array(self.config.mean, dtype=np.float32)
            self.std = np.array(self.config.std, dtype=np.float32)
        else:
            # Default to zero mean unit variance
            self.mean = np.zeros(self.config.input_channels, dtype=np.float32)
            self.std = np.ones(self.config.input_channels, dtype=np.float32)
    
    def preprocess_single(
        self,
        image: np.ndarray,
        bboxes: Optional[List[List[float]]] = None,
        return_metadata: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Preprocess a single image.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            bboxes: Optional bounding boxes for augmentation
            return_metadata: Whether to return preprocessing metadata
            
        Returns:
            Preprocessed image (and optional metadata)
        """
        start_time = time.time()
        
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Empty image provided")
            
            # Store original metadata
            metadata = {
                "original_shape": image.shape,
                "original_dtype": image.dtype,
                "original_range": (image.min(), image.max())
            }
            
            # Convert to appropriate format
            image = self._ensure_format(image)
            
            # Quality check
            quality_score = self._assess_image_quality(image)
            metadata["quality_score"] = quality_score
            
            if quality_score < self.config.quality_threshold:
                logger.warning("Low quality image detected (score: %.2f)", quality_score)
            
            # Apply preprocessing pipeline
            processed, transform_metadata = self._apply_preprocessing_pipeline(image, bboxes)
            metadata.update(transform_metadata)
            
            # Final validation
            processed = self._validate_output(processed)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.frame_count += 1
            self.quality_scores.append(quality_score)
            
            metadata["processing_time_ms"] = processing_time * 1000
            metadata["final_shape"] = processed.shape
            
            if self.config.debug_mode:
                self._debug_log(metadata, image, processed)
            
            if return_metadata:
                return processed, metadata
            else:
                return processed
                
        except Exception as e:
            logger.error("Preprocessing failed: %s", str(e))
            raise
    
    def preprocess_batch(
        self,
        images: List[np.ndarray],
        bboxes_list: Optional[List[List[List[float]]]] = None
    ) -> List[np.ndarray]:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of input images
            bboxes_list: Optional list of bounding boxes for each image
            
        Returns:
            List of preprocessed images
        """
        if bboxes_list is None:
            bboxes_list = [None] * len(images)
        
        processed_images = []
        
        # Process in parallel if configured
        if self.config.num_workers > 1 and len(images) > 1:
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = []
                for image, bboxes in zip(images, bboxes_list):
                    future = executor.submit(self.preprocess_single, image, bboxes, False)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        processed = future.result()
                        processed_images.append(processed)
                    except Exception as e:
                        logger.error("Batch preprocessing error: %s", str(e))
                        processed_images.append(None)
        else:
            # Sequential processing
            for image, bboxes in zip(images, bboxes_list):
                try:
                    processed = self.preprocess_single(image, bboxes, False)
                    processed_images.append(processed)
                except Exception as e:
                    logger.error("Batch preprocessing error: %s", str(e))
                    processed_images.append(None)
        
        return processed_images
    
    def _ensure_format(self, image: np.ndarray) -> np.ndarray:
        """
        Ensure image is in correct format and dimensions.
        
        Args:
            image: Input image
            
        Returns:
            Formatted image
        """
        # Convert to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Handle different input dimensions
        if len(image.shape) == 2:  # Grayscale (H, W)
            image = np.expand_dims(image, axis=-1)
            if self.config.input_channels == 3:
                # Convert grayscale to RGB
                image = np.repeat(image, 3, axis=-1)
        elif len(image.shape) == 3:
            if image.shape[-1] == 1 and self.config.input_channels == 3:
                # Expand single channel to RGB
                image = np.repeat(image, 3, axis=-1)
            elif image.shape[-1] == 3 and self.config.input_channels == 1:
                # Convert RGB to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.expand_dims(image, axis=-1)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # Ensure correct data type
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert color space if needed
        if self.config.input_format == ImageFormat.RGB and image.shape[-1] == 3:
            if image.shape[-1] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.config.input_format == ImageFormat.GRAY and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=-1)
        elif self.config.input_format == ImageFormat.HSV and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.config.input_format == ImageFormat.LAB and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif self.config.input_format == ImageFormat.YUV and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        return image
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """
        Assess image quality using multiple metrics.
        
        Args:
            image: Input image
            
        Returns:
            Quality score between 0 and 1
        """
        if image is None or image.size == 0:
            return 0.0
        
        quality_metrics = []
        
        try:
            # 1. Brightness assessment
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.squeeze()
            
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2.0  # Best at 0.5
            quality_metrics.append(brightness_score)
            
            # 2. Contrast assessment (using standard deviation)
            contrast = np.std(gray) / 128.0  # Normalized
            contrast_score = min(1.0, contrast)
            quality_metrics.append(contrast_score)
            
            # 3. Sharpness assessment (using Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            sharpness_normalized = min(1.0, sharpness / 1000.0)  # Empirical threshold
            quality_metrics.append(sharpness_normalized)
            
            # 4. Noise assessment (using wavelet denoising residual)
            # Simplified version using variance of difference from median filter
            median_filtered = cv2.medianBlur(gray, 3)
            noise_residual = np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))
            noise_level = np.mean(noise_residual) / 255.0
            noise_score = 1.0 - min(1.0, noise_level * 5.0)  # Lower noise = better
            quality_metrics.append(noise_score)
            
            # 5. Blur assessment (using variance of LoG)
            blur_radius = 3
            blurred = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)
            log = cv2.Laplacian(blurred, cv2.CV_64F)
            blur_variance = np.var(log)
            blur_score = min(1.0, blur_variance / 100.0)  # Empirical
            quality_metrics.append(blur_score)
            
            # 6. Saturation assessment (for color images)
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                saturation = np.mean(hsv[:, :, 1]) / 255.0
                saturation_score = min(1.0, saturation * 2.0)  # Some saturation is good
                quality_metrics.append(saturation_score)
            
            # Calculate weighted average
            weights = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]  # Weights for each metric
            weights = weights[:len(quality_metrics)]  # Adjust if fewer metrics
            
            # Normalize weights
            weights = np.array(weights) / np.sum(weights)
            
            quality_score = np.dot(quality_metrics[:len(weights)], weights)
            
            return float(quality_score)
            
        except Exception as e:
            logger.warning("Quality assessment failed: %s", str(e))
            return 0.5  # Default medium quality
    
    def _apply_preprocessing_pipeline(
        self,
        image: np.ndarray,
        bboxes: Optional[List[List[float]]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply the complete preprocessing pipeline.
        
        Args:
            image: Input image
            bboxes: Optional bounding boxes
            
        Returns:
            Tuple of (processed image, metadata)
        """
        metadata = {}
        processed = image.copy()
        
        # 1. Resize to target dimensions
        if processed.shape[:2] != (self.config.input_height, self.config.input_width):
            processed, resize_metadata = self._resize_image(processed, bboxes)
            metadata.update(resize_metadata)
        
        # 2. Apply augmentations (if enabled and bboxes provided)
        if self.config.enable_augmentation and self.augmentation_pipeline is not None:
            if bboxes is not None and len(bboxes) > 0:
                # Convert bboxes to albumentations format
                albu_bboxes = []
                for bbox in bboxes:
                    if len(bbox) >= 4:
                        # Format: [x_min, y_min, x_max, y_max, class_id]
                        if len(bbox) == 4:
                            albu_bboxes.append([*bbox, 0])  # Add default class_id
                        else:
                            albu_bboxes.append(bbox[:5])  # Use first 5 elements
                
                # Apply augmentation
                augmented = self.augmentation_pipeline(
                    image=processed,
                    bboxes=albu_bboxes
                )
                processed = augmented['image']
                
                # Update bboxes
                if bboxes is not None:
                    bboxes[:] = augmented['bboxes']
                
                metadata["augmentation_applied"] = True
            else:
                # Apply only image augmentation
                augmented = self.augmentation_pipeline(image=processed)
                processed = augmented['image']
                metadata["augmentation_applied"] = True
        
        # 3. Apply image enhancements
        if self.config.enable_histogram_equalization:
            processed = self._apply_histogram_equalization(processed)
            metadata["histogram_equalized"] = True
        
        if self.config.enable_denoising:
            processed = self._apply_denoising(processed)
            metadata["denoised"] = True
        
        if self.config.enable_sharpening:
            processed = self._apply_sharpening(processed)
            metadata["sharpened"] = True
        
        if self.config.enable_contrast_enhancement:
            processed = self._apply_contrast_enhancement(processed)
            metadata["contrast_enhanced"] = True
        
        if self.config.enable_color_correction:
            processed = self._apply_color_correction(processed)
            metadata["color_corrected"] = True
        
        # 4. Apply normalization
        processed = self._apply_normalization(processed)
        metadata["normalization_method"] = self.config.normalization_method.value
        
        return processed, metadata
    
    def _resize_image(
        self,
        image: np.ndarray,
        bboxes: Optional[List[List[float]]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image
            bboxes: Optional bounding boxes to resize
            
        Returns:
            Tuple of (resized image, metadata)
        """
        metadata = {
            "original_size": image.shape[:2],
            "target_size": (self.config.input_height, self.config.input_width)
        }
        
        original_h, original_w = image.shape[:2]
        target_h, target_w = self.config.input_height, self.config.input_width
        
        # Calculate scaling factors
        scale_x = target_w / original_w
        scale_y = target_h / original_h
        
        # Choose interpolation method based on scaling
        if scale_x < 1.0 or scale_y < 1.0:  # Downscaling
            interpolation = cv2.INTER_AREA
        else:  # Upscaling
            interpolation = cv2.INTER_CUBIC
        
        # Resize image
        resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        
        # Resize bounding boxes if provided
        if bboxes is not None:
            for i, bbox in enumerate(bboxes):
                if len(bbox) >= 4:
                    # Resize coordinates
                    bbox[0] *= scale_x  # x_min
                    bbox[1] *= scale_y  # y_min
                    bbox[2] *= scale_x  # x_max
                    bbox[3] *= scale_y  # y_max
                    
                    # Clip to image boundaries
                    bbox[0] = max(0, min(bbox[0], target_w - 1))
                    bbox[1] = max(0, min(bbox[1], target_h - 1))
                    bbox[2] = max(0, min(bbox[2], target_w - 1))
                    bbox[3] = max(0, min(bbox[3], target_h - 1))
        
        metadata.update({
            "scale_x": scale_x,
            "scale_y": scale_y,
            "interpolation": interpolation
        })
        
        return resized, metadata
    
    def _apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization."""
        if len(image.shape) == 3:
            # Apply CLAHE to each channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            channels = cv2.split(image)
            equalized_channels = []
            
            for channel in channels:
                equalized_channels.append(clahe.apply(channel))
            
            return cv2.merge(equalized_channels)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def _apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising filter."""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            return cv2.fastNlMeansDenoising(
                image,
                None,
                h=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
    
    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening filter."""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def _apply_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast enhancement."""
        # Convert to LAB color space for better contrast enhancement
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            return exposure.equalize_adapthist(image, clip_limit=0.03)
    
    def _apply_color_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply color correction."""
        # Simple white balance using gray world assumption
        if len(image.shape) == 3:
            result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            avg_a = np.average(result[:, :, 1])
            avg_b = np.average(result[:, :, 2])
            result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
            result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
            return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return image
    
    def _apply_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply normalization to image.
        
        Args:
            image: Input image (uint8)
            
        Returns:
            Normalized image (float32)
        """
        # Convert to float32
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # Apply normalization based on method
        if self.config.normalization_method == NormalizationMethod.STANDARD:
            # (x - mean) / std
            if len(image.shape) == 3:
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
            else:
                image = (image - self.mean[0]) / self.std[0]
        
        elif self.config.normalization_method == NormalizationMethod.MINMAX:
            # (x - min) / (max - min)
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
        
        elif self.config.normalization_method == NormalizationMethod.ZERO_MEAN:
            # x - mean
            if len(image.shape) == 3:
                for i in range(3):
                    image[:, :, i] = image[:, :, i] - self.mean[i]
            else:
                image = image - self.mean[0]
        
        elif self.config.normalization_method == NormalizationMethod.UNIT_STD:
            # x / std
            if len(image.shape) == 3:
                for i in range(3):
                    image[:, :, i] = image[:, :, i] / self.std[i]
            else:
                image = image / self.std[0]
        
        elif self.config.normalization_method == NormalizationMethod.PER_CHANNEL:
            # Normalize each channel independently
            if len(image.shape) == 3:
                for i in range(3):
                    channel = image[:, :, i]
                    channel_mean = channel.mean()
                    channel_std = channel.std()
                    if channel_std > 0:
                        image[:, :, i] = (channel - channel_mean) / channel_std
            else:
                channel_mean = image.mean()
                channel_std = image.std()
                if channel_std > 0:
                    image = (image - channel_mean) / channel_std
        
        elif self.config.normalization_method == NormalizationMethod.IMAGENET:
            # ImageNet normalization
            if len(image.shape) == 3:
                image[:, :, 0] = (image[:, :, 0] - self.mean[0]) / self.std[0]
                image[:, :, 1] = (image[:, :, 1] - self.mean[1]) / self.std[1]
                image[:, :, 2] = (image[:, :, 2] - self.mean[2]) / self.std[2]
            else:
                image = (image - self.mean[0]) / self.std[0]
        
        return image
    
    def _validate_output(self, image: np.ndarray) -> np.ndarray:
        """
        Validate preprocessed image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Validated image
        """
        # Check for NaN or Inf values
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            logger.error("Image contains NaN or Inf values")
            # Replace with zeros
            image = np.nan_to_num(image)
        
        # Check value range (for normalized images)
        if image.dtype == np.float32:
            # Allow some tolerance for normalized values
            if np.abs(image).max() > 10.0:  # Arbitrary threshold
                logger.warning("Image values out of expected range")
        
        return image
    
    def _debug_log(self, metadata: Dict[str, Any], original: np.ndarray, processed: np.ndarray):
        """Log debug information."""
        logger.debug("Preprocessing metadata: %s", metadata)
        
        if self.config.save_preprocessed:
            import os
            os.makedirs(self.config.save_path, exist_ok=True)
            
            # Save original and processed images
            timestamp = int(time.time() * 1000)
            orig_path = os.path.join(self.config.save_path, f"original_{timestamp}.png")
            proc_path = os.path.join(self.config.save_path, f"processed_{timestamp}.png")
            
            # Convert back to uint8 for saving
            if original.dtype != np.uint8:
                orig_save = (original * 255).astype(np.uint8) if original.max() <= 1.0 else original.astype(np.uint8)
            else:
                orig_save = original
            
            if processed.dtype != np.uint8:
                # Denormalize if needed
                if self.config.normalization_method == NormalizationMethod.IMAGENET:
                    proc_save = processed.copy()
                    if len(proc_save.shape) == 3:
                        for i in range(3):
                            proc_save[:, :, i] = proc_save[:, :, i] * self.std[i] + self.mean[i]
                    else:
                        proc_save = proc_save * self.std[0] + self.mean[0]
                    proc_save = (proc_save * 255).clip(0, 255).astype(np.uint8)
                else:
                    proc_save = (processed * 255).clip(0, 255).astype(np.uint8)
            else:
                proc_save = processed
            
            cv2.imwrite(orig_path, orig_save)
            cv2.imwrite(proc_path, proc_save)
            
            logger.debug("Saved images to %s", self.config.save_path)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.processing_times:
            return {}
        
        return {
            "total_frames_processed": self.frame_count,
            "avg_processing_time_ms": np.mean(self.processing_times) * 1000,
            "std_processing_time_ms": np.std(self.processing_times) * 1000,
            "max_processing_time_ms": np.max(self.processing_times) * 1000,
            "min_processing_time_ms": np.min(self.processing_times) * 1000,
            "avg_quality_score": np.mean(self.quality_scores) if self.quality_scores else 0.0,
            "current_fps": 1000.0 / (np.mean(self.processing_times) * 1000) if self.processing_times else 0.0
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.processing_times = []
        self.frame_count = 0
        self.quality_scores = []
        logger.info("Performance statistics reset")


# Factory function for creating preprocessors
def create_preprocessor(config: Dict[str, Any] = None) -> ImagePreprocessor:
    """
    Factory function to create an ImagePreprocessor instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ImagePreprocessor instance
    """
    if config is None:
        config = {}
    
    # Convert dict to PreprocessingConfig
    config_obj = PreprocessingConfig(**config)
    return ImagePreprocessor(config_obj)


# Utility functions for common preprocessing tasks
def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = False,
    padding_value: int = 114
) -> np.ndarray:
    """
    Resize image to target dimensions.
    
    Args:
        image: Input image
        target_size: (width, height) target dimensions
        keep_aspect_ratio: Whether to maintain aspect ratio
        padding_value: Padding value for letterboxing
        
    Returns:
        Resized image
    """
    if image is None or image.size == 0:
        return image
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    if keep_aspect_ratio:
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # New dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to target size
        dw = target_w - new_w
        dh = target_h - new_h
        
        top = dh // 2
        bottom = dh - top
        left = dw // 2
        right = dw - left
        
        # Add padding
        if len(image.shape) == 3:
            padding = ((top, bottom), (left, right), (0, 0))
        else:
            padding = ((top, bottom), (left, right))
        
        resized = np.pad(resized, padding, mode='constant', constant_values=padding_value)
        
        return resized
    else:
        # Simple resize without aspect ratio preservation
        return cv2.resize(image, target_size)


def normalize_image(
    image: np.ndarray,
    mean: Union[float, Tuple[float, float, float]] = (0.485, 0.456, 0.406),
    std: Union[float, Tuple[float, float, float]] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize image using mean and std.
    
    Args:
        image: Input image (float32)
        mean: Mean values
        std: Standard deviation values
        
    Returns:
        Normalized image
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    if isinstance(mean, (int, float)):
        mean = (mean,) * 3
    if isinstance(std, (int, float)):
        std = (std,) * 3
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image[:, :, 0] = (image[:, :, 0] - mean[0]) / std[0]
        image[:, :, 1] = (image[:, :, 1] - mean[1]) / std[1]
        image[:, :, 2] = (image[:, :, 2] - mean[2]) / std[2]
    else:
        image = (image - mean[0]) / std[0]
    
    return image


def denormalize_image(
    image: np.ndarray,
    mean: Union[float, Tuple[float, float, float]] = (0.485, 0.456, 0.406),
    std: Union[float, Tuple[float, float, float]] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Denormalize image using mean and std.
    
    Args:
        image: Normalized image (float32)
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        Denormalized image (uint8)
    """
    if isinstance(mean, (int, float)):
        mean = (mean,) * 3
    if isinstance(std, (int, float)):
        std = (std,) * 3
    
    denormalized = image.copy()
    
    if len(denormalized.shape) == 3 and denormalized.shape[2] == 3:
        denormalized[:, :, 0] = denormalized[:, :, 0] * std[0] + mean[0]
        denormalized[:, :, 1] = denormalized[:, :, 1] * std[1] + mean[1]
        denormalized[:, :, 2] = denormalized[:, :, 2] * std[2] + mean[2]
    else:
        denormalized = denormalized * std[0] + mean[0]
    
    # Convert to uint8
    denormalized = (denormalized * 255).clip(0, 255).astype(np.uint8)
    
    return denormalized


def apply_augmentation(
    image: np.ndarray,
    augmentation_type: str = "light",
    bboxes: Optional[List[List[float]]] = None
) -> Union[np.ndarray, Tuple[np.ndarray, List[List[float]]]]:
    """
    Apply standard augmentation to image.
    
    Args:
        image: Input image
        augmentation_type: Type of augmentation ('light', 'medium', 'heavy')
        bboxes: Optional bounding boxes
        
    Returns:
        Augmented image (and optional bboxes)
    """
    if augmentation_type == "light":
        transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']) if bboxes else None)
    elif augmentation_type == "medium":
        transforms = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']) if bboxes else None)
    elif augmentation_type == "heavy":
        transforms = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.5),
            A.GaussianBlur(blur_limit=(5, 9), p=0.3),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']) if bboxes else None)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    if bboxes is not None:
        class_labels = [bbox[4] if len(bbox) > 4 else 0 for bbox in bboxes]
        bboxes_no_labels = [bbox[:4] for bbox in bboxes]
        
        augmented = transforms(
            image=image,
            bboxes=bboxes_no_labels,
            class_labels=class_labels
        )
        
        # Recombine bboxes with labels
        augmented_bboxes = []
        for bbox, label in zip(augmented['bboxes'], augmented['class_labels']):
            augmented_bboxes.append([*bbox, label])
        
        return augmented['image'], augmented_bboxes
    else:
        augmented = transforms(image=image)
        return augmented['image']


def compute_image_statistics(image: np.ndarray) -> Dict[str, Any]:
    """
    Compute comprehensive image statistics.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary of statistics
    """
    if image is None or image.size == 0:
        return {}
    
    stats = {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "min_value": float(image.min()),
        "max_value": float(image.max()),
        "mean_value": float(image.mean()),
        "std_value": float(image.std()),
    }
    
    if len(image.shape) == 3:
        for i in range(image.shape[2]):
            channel = image[:, :, i]
            stats[f"channel_{i}_mean"] = float(channel.mean())
            stats[f"channel_{i}_std"] = float(channel.std())
            stats[f"channel_{i}_min"] = float(channel.min())
            stats[f"channel_{i}_max"] = float(channel.max())
    
    # Histogram statistics
    if image.dtype == np.uint8:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        stats["histogram_entropy"] = float(-np.sum(hist * np.log(hist + 1e-10)))
    
    return stats
