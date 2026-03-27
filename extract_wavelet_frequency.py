import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import pywt
import csv
from tqdm import tqdm
import logging
import gc
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_orthogonal(matrix, atol=1e-4):
    """检查矩阵是否正交（列向量正交）"""
    identity = np.eye(matrix.shape[0])
    return np.allclose(matrix.T @ matrix, identity, atol=atol, rtol=0)

def extract_frequency_features(amplitude_spectrum, image_spacing=None, image_direction=None):
    """提取频域特征（修正空间参数和方向矩阵处理）"""
    if np.sum(amplitude_spectrum) == 0:
        logger.warning("空的振幅谱，返回默认特征")
        return {k: 0 for k in [
            'mean_amplitude', 'std_amplitude', 'max_amplitude', 'min_amplitude', 'total_energy',
            'power_spectral_density', 'contrast', 'centroid_x', 'centroid_y', 'centroid_z',
            'centroid_avg', 'spread_x', 'spread_y', 'spread_z', 'spread_avg', 'entropy'
        ]}

    features = {}
    eps = 1e-10

    # ====================== 基础维度参数 ======================
    Nz, Ny, Nx = amplitude_spectrum.shape
    spacing = image_spacing if image_spacing else (1.0, 1.0, 1.0)
    spacing_x, spacing_y, spacing_z = spacing
    voxel_volume = spacing_x * spacing_y * spacing_z

    # ====================== 基本统计特征 ======================
    features['mean_amplitude'] = np.mean(amplitude_spectrum)
    features['std_amplitude'] = np.std(amplitude_spectrum)
    features['max_amplitude'] = np.max(amplitude_spectrum)
    features['min_amplitude'] = np.min(amplitude_spectrum)
    features['contrast'] = features['max_amplitude'] - features['min_amplitude']

    # ====================== 总能量（功率谱积分，线性频率密度） ======================
    power_spectrum = amplitude_spectrum ** 2
    total_power = np.sum(power_spectrum)
    features['total_energy'] = total_power * voxel_volume / (Nx * Ny * Nz)

    # ====================== 频率分辨率（线性频率，无2π因子） ======================
    dkx = 1.0 / (Nx * spacing_x)
    dky = 1.0 / (Ny * spacing_y)
    dkz = 1.0 / (Nz * spacing_z)
    freq_resolution = dkx * dky * dkz

    # ====================== 功率谱密度（PSD，线性频率密度） ======================
    freq_volume_x = Nx * dkx
    freq_volume_y = Ny * dky
    freq_volume_z = Nz * dkz
    total_freq_volume = freq_volume_x * freq_volume_y * freq_volume_z
    features['power_spectral_density'] = total_power / total_freq_volume if total_power > eps else 0.0

    # ====================== 频率网格生成（列优先方向矩阵） ======================
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=spacing_x))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=spacing_y))
    fz = np.fft.fftshift(np.fft.fftfreq(Nz, d=spacing_z))
    fz_grid, fy_grid, fx_grid = np.meshgrid(fz, fy, fx, indexing='ij')
    dir_matrix = np.eye(3)  # 默认单位矩阵
    if image_direction is not None:
        dir_list = np.array(image_direction)
        dir_matrix = dir_list.reshape(3, 3, order='C')
        if not is_orthogonal(dir_matrix, atol=1e-3):
            logger.warning("方向矩阵非正交，强制转换为单位矩阵")
            dir_matrix = np.eye(3)

    # ====================== 物理频率分量计算（修正方向矩阵应用） ======================
    fx_phys = fx_grid * dir_matrix[0, 0] + fy_grid * dir_matrix[0, 1] + fz_grid * dir_matrix[0, 2]
    fy_phys = fx_grid * dir_matrix[1, 0] + fy_grid * dir_matrix[1, 1] + fz_grid * dir_matrix[1, 2]
    fz_phys = fx_grid * dir_matrix[2, 0] + fy_grid * dir_matrix[2, 1] + fz_grid * dir_matrix[2, 2]
    # ====================== 质心计算（功率谱加权平均） ======================
    if total_power > eps:
        centroid_x = np.sum(fx_phys * power_spectrum) / total_power
        centroid_y = np.sum(fy_phys * power_spectrum) / total_power
        centroid_z = np.sum(fz_phys * power_spectrum) / total_power
    else:
        centroid_x = centroid_y = centroid_z = 0.0
    features.update({
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'centroid_z': centroid_z,
        'centroid_avg': (centroid_x + centroid_y + centroid_z) / 3
    })

    # ====================== 扩展度计算（标准差，功率谱加权） ======================
    if total_power > eps:
        spread_x = np.sqrt(np.sum((fx_phys - centroid_x) ** 2 * power_spectrum) / total_power)
        spread_y = np.sqrt(np.sum((fy_phys - centroid_y) ** 2 * power_spectrum) / total_power)
        spread_z = np.sqrt(np.sum((fz_phys - centroid_z) ** 2 * power_spectrum) / total_power)
    else:
        spread_x = spread_y = spread_z = 0.0
    features.update({
        'spread_x': spread_x,
        'spread_y': spread_y,
        'spread_z': spread_z,
        'spread_avg': (spread_x + spread_y + spread_z) / 3
    })

    # ====================== 熵计算（信息论熵，基于功率谱概率分布） ======================
    if total_power > eps:
        prob = power_spectrum / total_power
        features['entropy'] = -np.sum(prob * np.log2(prob + eps))  # 避免log2(0)
    else:
        features['entropy'] = 0.0
    return features
def zero_wavelet_details(coeffs):
    """递归置零三维小波细节系数（仅保留低频近似）"""
    if isinstance(coeffs, (list, tuple)):
        for i, c in enumerate(coeffs):
            if i == 0:  # 保留低频系数（索引0为近似系数）
                continue
            if isinstance(c, (np.ndarray, list, tuple)):
                if isinstance(c, np.ndarray):
                    c.fill(0)  # 直接置零数组
                else:
                    zero_wavelet_details(c)  # 递归处理子系数
            elif isinstance(c, dict):
                for key in c:
                    zero_wavelet_details(c[key])
def adjust_to_wavelet_size(image, wavelet_name, level=1, pad_mode='reflect', **pad_kwargs):
    """调整图像尺寸并返回填充后的图像及填充宽度"""
    try:
        new_shape = []
        for dim in image.shape:
            divisor = 2 ** level
            remainder = dim % divisor
            new_dim = dim if remainder == 0 else dim + (divisor - remainder)
            new_shape.append(new_dim)
        pad_width = []
        for dim, new_dim in zip(image.shape, new_shape):
            pad_total = new_dim - dim
            pad_before = (pad_total + 1) // 2
            pad_after = pad_total - pad_before
            pad_width.append((pad_before, pad_after))
        padded_image = np.pad(image, pad_width, mode=pad_mode, **pad_kwargs)
        return padded_image, pad_width
    except Exception as e:
        logger.error(f"调整小波尺寸失败: {str(e)}")
        return image, None

def process_wavelet(image, wavelet_name, roi_mask, level=1):
    """处理小波变换并正确裁剪"""
    try:
        roi_mask_bool = roi_mask.astype(bool)
        masked_image = image * roi_mask_bool.astype(np.float32)
        original_shape = image.shape
        adjusted, pad_width = adjust_to_wavelet_size(masked_image, wavelet_name, level, pad_mode='reflect')
        if pad_width is None:
            return None
        # 调整ROI使用相同的填充参数
        adjusted_roi, _ = adjust_to_wavelet_size(roi_mask_bool.astype(np.float32), wavelet_name, level,
                                                 pad_mode='constant', constant_values=0)
        adjusted_roi = adjusted_roi.astype(bool)
        coeffs = pywt.wavedecn(adjusted, wavelet_name, level=level, mode='symmetric')
        zero_wavelet_details(coeffs)
        reconstructed = pywt.waverecn(coeffs, wavelet_name, mode='symmetric')
        # 正确裁剪填充部分
        crop_slices = [
            slice(pad[0], pad[0] + original_dim)
            for pad, original_dim in zip(pad_width, original_shape)
        ]
        cropped = reconstructed[tuple(crop_slices)]
        result = cropped * roi_mask_bool.astype(np.float32)
        return result
    except Exception as e:
        logger.error(f"小波处理失败: {str(e)}", exc_info=True)
        return None
def validate_image_roi(image, roi):
    if image.GetDimension() != 3 or roi.GetDimension() != 3:
        logger.error("图像或ROI非三维，必须为3D数据")
        return False

    if image.GetSize() != roi.GetSize():
        logger.error("图像与ROI尺寸不匹配")
        return False

    if not np.allclose(image.GetSpacing(), roi.GetSpacing(), atol=1e-4):
        logger.error("间距参数不匹配")
        return False

    if not np.allclose(image.GetOrigin(), roi.GetOrigin(), atol=1e-4):
        logger.error("原点参数不匹配")
        return False
    img_dir_list = np.array(image.GetDirection())
    roi_dir_list = np.array(roi.GetDirection())
    img_dir = img_dir_list.reshape(3, 3, order='C')
    roi_dir = roi_dir_list.reshape(3, 3, order='C')
    if not (is_orthogonal(img_dir) and is_orthogonal(roi_dir)):
        logger.error("方向矩阵非正交")
        return False
    if not np.allclose(img_dir, roi_dir, atol=1e-4):
        logger.error("方向矩阵不一致")
        return False
    roi_array = sitk.GetArrayFromImage(roi).astype(np.int32)
    if not np.isin(np.unique(roi_array), [0, 1]).all():
        logger.error("ROI必须为0/1二值图像")
        return False
    if roi_array.sum() < 10:
        logger.warning("ROI体积过小，可能影响特征计算")
    return True
def main_processing(image_folder, roi_folder, output_csv, log_sigmas, wavelet_types):
    features_list = []
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    for img_file in tqdm(os.listdir(image_folder)):
        if not img_file.endswith(('.nii', '.nii.gz')):
            continue
        try:
            img_path = os.path.join(image_folder, img_file)
            roi_path = os.path.join(roi_folder, img_file)
            if not os.path.exists(roi_path):
                logger.warning(f"跳过缺失ROI的文件: {img_file}")
                continue
            logger.info(f"处理文件: {img_file}")
            image = sitk.ReadImage(img_path)
            roi = sitk.ReadImage(roi_path)

            if not validate_image_roi(image, roi):
                logger.warning(f"空间信息不兼容，跳过: {img_file}")
                continue
            spacing = image.GetSpacing()
            direction = image.GetDirection()
            img_np = sitk.GetArrayFromImage(image).astype(np.float32)
            roi_np = sitk.GetArrayFromImage(roi).astype(bool)

            if roi_np.sum() == 0:
                logger.warning(f"ROI为空，跳过: {img_file}")
                continue
            row = {'patient_id': os.path.splitext(img_file)[0]}
            # ----------------------- 原始图像特征 -----------------------
            logger.info(f"计算原始图像频域特征: {img_file}")
            masked_orig = img_np * roi_np
            fft_orig = np.fft.fftn(masked_orig)
            amp_orig = np.abs(np.fft.fftshift(fft_orig))
            row.update({f'raw_{k}': v for k, v in
                        extract_frequency_features(amp_orig, spacing, direction).items()})

            del fft_orig, amp_orig, masked_orig
            gc.collect()

            # ----------------------- LoG特征处理 -----------------------
            for sigma in log_sigmas:
                sigma_pixels = (
                    sigma / spacing[2],
                    sigma / spacing[1],
                    sigma / spacing[0]
                )
                logger.info(f"计算LoG特征 (sigma={sigma}): {img_file}")
                log_img = ndimage.gaussian_laplace(img_np, sigma=sigma_pixels)
                log_masked = log_img * roi_np

                fft_log = np.fft.fftn(log_masked)
                amp_log = np.abs(np.fft.fftshift(fft_log))
                row.update({f'log_{sigma:.1f}_{k}': v for k, v in
                            extract_frequency_features(amp_log, spacing, direction).items()})

                del log_img, log_masked, fft_log, amp_log
                gc.collect()

            # ----------------------- 小波特征处理 -----------------------
            for wavelet in wavelet_types:
                logger.info(f"计算小波特征 ({wavelet}): {img_file}")
                approx = process_wavelet(img_np, wavelet, roi_np)
                if approx is None:
                    logger.warning(f"小波处理失败，跳过: {wavelet}")
                    continue

                fft_wave = np.fft.fftn(approx)
                amp_wave = np.abs(np.fft.fftshift(fft_wave))
                row.update({f'wave_{wavelet}_{k}': v for k, v in
                            extract_frequency_features(amp_wave, spacing, direction).items()})

                del approx, fft_wave, amp_wave
                gc.collect()

            features_list.append(row)
            del img_np, roi_np
            gc.collect()

        except Exception as e:
            logger.error(f"处理失败 [{img_file}]: {str(e)}", exc_info=True)
            gc.collect()
            continue

    if features_list:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=features_list[0].keys())
            writer.writeheader()
            writer.writerows(features_list)
        logger.info(f"成功保存 {len(features_list)} 条特征到 {output_csv}")
    else:
        logger.warning("未生成有效特征数据")


if __name__ == "__main__":
    # 配置参数（根据实际路径修改）
    CONFIG = {
        'image_folder': r'D:\4',  # 原始图像文件夹（NIfTI格式）
        'roi_folder': r'D:\3',  # ROI掩码文件夹（同名NIfTI，二值图像）
        'output_csv': r'D:\frequency_features3.csv',  # 输出CSV路径
        'log_sigmas': [1.0, 2.0, 3.0],  # LoG滤波器的Sigma值
        'wavelet_types': ['haar', 'db2']  # 支持的小波类型（需满足pywt）
    }

    main_processing(**CONFIG)