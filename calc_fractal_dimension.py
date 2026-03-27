import os
import numpy as np
import nibabel as nib
from scipy import stats
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import zoom
# multiprocessing and joblib are already imported in the original code, but listing them explicitly for clarity
import multiprocessing
from joblib import Parallel, delayed


# --- Core Functions ---

def resample_to_isotropic(data, original_spacing, target_spacing=1.0):
    """
    将3D体数据重采样到各向同性分辨率 (using nearest-neighbor interpolation)
    Args:
        data (np.ndarray): 输入的3D numpy数组 (预期 Z, Y, X 顺序).
        original_spacing (list or tuple): 原始各轴的体素间距 (预期 dz, dy, dx).
        target_spacing (float): 目标各向同性体素间距.
    Returns:
        np.ndarray: 重采样后的3D numpy数组.
    """
    # Calculate zoom factors based on desired spacing
    zoom_factors = [orig / target_spacing for orig in original_spacing]
    # Use nearest-neighbor interpolation (order=0) to preserve binary characteristics
    resampled_data = zoom(data, zoom_factors, order=0, mode='nearest')
    return resampled_data


def crop_zeros_3d(mask_3d):
    """
    裁剪掉三维mask在x、y、z三个方向上的全零切片.
    Args:
        mask_3d (np.ndarray): 输入的3D numpy数组 (预期 Z, Y, X 顺序).
    Returns:
        tuple: (cropped_mask, boundaries) or (None, None) if input is empty or cannot be cropped.
               boundaries is a dictionary containing min/max indices and dimensions.
    """
    # Check Z-dimension for non-zero slices
    z_nonzero = np.any(mask_3d, axis=(1, 2))  # Check across YX plane
    if not np.any(z_nonzero): return None, None  # Return None if all slices are zero
    z_indices = np.where(z_nonzero)[0]
    z_min = z_indices[0]
    z_max = z_indices[-1] + 1  # Slice index is exclusive at the end

    # Check Y-dimension for non-zero slices
    y_nonzero = np.any(mask_3d, axis=(0, 2))  # Check across ZX plane
    if not np.any(y_nonzero): return None, None
    y_indices = np.where(y_nonzero)[0]
    y_min = y_indices[0]
    y_max = y_indices[-1] + 1

    # Check X-dimension for non-zero slices
    x_nonzero = np.any(mask_3d, axis=(0, 1))  # Check across ZY plane
    if not np.any(x_nonzero): return None, None
    x_indices = np.where(x_nonzero)[0]
    x_min = x_indices[0]
    x_max = x_indices[-1] + 1

    # Crop the mask using the determined boundaries
    cropped_mask = mask_3d[z_min:z_max, y_min:y_max, x_min:x_max]

    # Store boundary information (optional, useful for debugging)
    boundaries = {
        'z_min': z_min, 'z_max': z_max, 'z_dim': z_max - z_min,
        'y_min': y_min, 'y_max': y_max, 'y_dim': y_max - y_min,
        'x_min': x_min, 'x_max': x_max, 'x_dim': x_max - x_min
    }

    # Return None if the cropped mask is empty
    if cropped_mask.size == 0:
        return None, None

    return cropped_mask, boundaries


def calculate_fd_3d(mask_3d, voxel_size=1.0, num_offsets=5):
    """
    Calculate the 3D Fractal Dimension (FD) using the box-counting method.
    Uses physical dimensions (mm) and incorporates grid offsets to reduce bias.
    Args:
        mask_3d (np.ndarray): The 3D binary mask (Z, Y, X).
        voxel_size (float): The physical size of each voxel in mm (isotropic).
        num_offsets (int): Number of random offsets to apply for stability.
    Returns:
        float: The calculated 3D Fractal Dimension, or np.nan if calculation fails.
    """
    # Basic validation
    if mask_3d is None or not np.any(mask_3d): return np.nan
    depth, height, width = mask_3d.shape
    if depth == 0 or height == 0 or width == 0: return np.nan

    # Calculate physical dimensions in mm
    physical_size_x = width * voxel_size
    physical_size_y = height * voxel_size
    physical_size_z = depth * voxel_size

    # Find the maximum physical dimension to determine the range of box sizes
    current_max_dim_size = max(physical_size_x, physical_size_y, physical_size_z)

    # If all physical dimensions are zero, return NaN
    if current_max_dim_size == 0: return np.nan
    # Ensure voxel_size is positive
    if voxel_size <= 0: return np.nan

    # Generate a list of box sizes (L) in powers of 2, starting from voxel_size
    L_list = []
    L = voxel_size
    temp_L = L
    while temp_L <= current_max_dim_size:
        L_list.append(temp_L)
        next_L = temp_L * 2
        # Prevent infinite loops or issues with large numbers/precision
        if next_L <= temp_L: break
        temp_L = next_L
        # Add a condition to avoid excessively large gaps if dimensions are very different
        if len(L_list) > 0 and temp_L / L_list[-1] < 1.5 and temp_L > current_max_dim_size: break

    # Need at least two scales for linear regression
    if len(L_list) < 2: return np.nan

    # Perform box counting with multiple offsets
    N_list_sum = np.zeros(len(L_list))  # Sum of counts across offsets for each L
    for offset_idx in range(num_offsets):
        # Generate random offset within [0, voxel_size) for each dimension
        offset_phys = np.random.rand(3) * voxel_size
        current_offset_N = np.zeros(len(L_list))  # Counts for the current offset

        for idx, L in enumerate(L_list):
            # Calculate the number of boxes needed along each dimension
            nx_boxes = int(np.ceil(physical_size_x / L)) if physical_size_x > 0 else 0
            ny_boxes = int(np.ceil(physical_size_y / L)) if physical_size_y > 0 else 0
            nz_boxes = int(np.ceil(physical_size_z / L)) if physical_size_z > 0 else 0

            count = 0
            # Iterate through all possible box starting positions
            for k in range(nz_boxes):
                for j in range(ny_boxes):
                    for i in range(nx_boxes):
                        # Calculate physical boundaries of the current box [start, end)
                        x_start_phys = i * L + offset_phys[0]
                        y_start_phys = j * L + offset_phys[1]
                        z_start_phys = k * L + offset_phys[2]

                        # Clip the end boundary to the physical size of the mask
                        x_end_phys = min(x_start_phys + L, physical_size_x)
                        y_end_phys = min(y_start_phys + L, physical_size_y)
                        z_end_phys = min(z_start_phys + L, physical_size_z)

                        # Skip invalid boxes (where end <= start, possible due to clipping/offset)
                        if x_end_phys <= x_start_phys or y_end_phys <= y_start_phys or z_end_phys <= z_start_phys:
                            continue

                        # Convert physical box boundaries to voxel indices [start_vox, end_vox)
                        # Using floor/ceil ensures coverage
                        x_start_vox = int(np.floor(x_start_phys / voxel_size))
                        x_end_vox = int(np.ceil(x_end_phys / voxel_size))
                        y_start_vox = int(np.floor(y_start_phys / voxel_size))
                        y_end_vox = int(np.ceil(y_end_phys / voxel_size))
                        z_start_vox = int(np.floor(z_start_phys / voxel_size))
                        z_end_vox = int(np.ceil(z_end_phys / voxel_size))

                        # Clip voxel indices to the mask's actual dimensions (0 to dim-1)
                        x_start_vox = max(0, x_start_vox)
                        x_end_vox = min(width, x_end_vox)
                        y_start_vox = max(0, y_start_vox)
                        y_end_vox = min(height, y_end_vox)
                        z_start_vox = max(0, z_start_vox)
                        z_end_vox = min(depth, z_end_vox)

                        # Check if the voxel range is valid before accessing the array
                        if x_end_vox > x_start_vox and y_end_vox > y_start_vox and z_end_vox > z_start_vox:
                            # Extract the box from the mask
                            box = mask_3d[z_start_vox:z_end_vox, y_start_vox:y_end_vox, x_start_vox:x_end_vox]
                            # Increment count if the box contains any part of the structure (any non-zero voxel)
                            if np.any(box):
                                count += 1

            current_offset_N[idx] = count  # Store count for this scale and offset

        N_list_sum += current_offset_N  # Add counts from this offset to the total sum

    # Calculate the average count N for each scale L across all offsets
    N_list = N_list_sum / num_offsets

    # --- Perform Linear Regression ---
    # Filter out scales where the count N is zero (log(0) is undefined)
    valid_mask = (N_list > 0)

    # Get the corresponding L values for valid counts
    L_list_filtered = np.array(L_list)[valid_mask]

    # Check if we still have enough points after filtering
    if len(L_list_filtered) < 2: return np.nan

    # Calculate log(L) and log(N)
    logL = np.log(L_list_filtered)
    logN = np.log(N_list[valid_mask])

    # Ensure log values are finite (should be, but good practice)
    finite_mask = np.isfinite(logL) & np.isfinite(logN)
    logL_finite = logL[finite_mask]
    logN_finite = logN[finite_mask]

    # Check again if we have enough finite points
    if len(logL_finite) < 2: return np.nan

    # Perform linear regression: logN = intercept + slope * logL
    slope, intercept, r_value, p_value, std_err = stats.linregress(logL_finite, logN_finite)

    # Fractal Dimension D is the negative of the slope
    # D = -slope
    # Ensure FD is non-negative if slope is negative (which is unusual)
    fd_result = -slope
    # return fd_result if fd_result >= 0 else np.nan # Optional: enforce non-negativity
    return fd_result


def calculate_fd_2d(slice_2d, voxel_size=1.0, num_offsets=3):
    """
    Calculate the 2D Fractal Dimension (FD) for a single slice.
    Uses physical dimensions (mm) and incorporates grid offsets.
    Args:
        slice_2d (np.ndarray): The 2D binary slice (expected Y, X).
        voxel_size (float): The physical size of each voxel in mm (isotropic).
        num_offsets (int): Number of random offsets to apply.
    Returns:
        float: The calculated 2D Fractal Dimension, or np.nan if calculation fails.
    """
    # Basic validation
    if slice_2d is None or not np.any(slice_2d): return np.nan
    height, width = slice_2d.shape  # Slice dimensions (Y, X)
    if height == 0 or width == 0: return np.nan

    # Calculate physical dimensions in mm
    physical_size_y = height * voxel_size
    physical_size_x = width * voxel_size

    current_max_dim_size = max(physical_size_x, physical_size_y)
    if current_max_dim_size == 0: return np.nan
    if voxel_size <= 0: return np.nan

    # Generate list of box sizes (L)
    L_list = []
    L = voxel_size
    temp_L = L
    while temp_L <= current_max_dim_size:
        L_list.append(temp_L)
        next_L = temp_L * 2
        if next_L <= temp_L: break
        temp_L = next_L
        if len(L_list) > 0 and temp_L / L_list[-1] < 1.5 and temp_L > current_max_dim_size: break

    if len(L_list) < 2: return np.nan

    # Perform box counting with multiple offsets
    N_list_sum = np.zeros(len(L_list))
    for offset_idx in range(num_offsets):
        offset_phys = np.random.rand(2) * voxel_size  # XY offsets
        current_offset_N = np.zeros(len(L_list))

        for idx, L in enumerate(L_list):
            # Calculate number of boxes needed
            nx_boxes = int(np.ceil(physical_size_x / L)) if physical_size_x > 0 else 0
            ny_boxes = int(np.ceil(physical_size_y / L)) if physical_size_y > 0 else 0

            count = 0
            # Iterate through possible box starting positions
            for j in range(ny_boxes):
                for i in range(nx_boxes):
                    # Calculate physical boundaries [start, end)
                    x_start_phys = i * L + offset_phys[0]
                    y_start_phys = j * L + offset_phys[1]

                    x_end_phys = min(x_start_phys + L, physical_size_x)
                    y_end_phys = min(y_start_phys + L, physical_size_y)

                    # Skip invalid boxes
                    if x_end_phys <= x_start_phys or y_end_phys <= y_start_phys:
                        continue

                    # Convert to voxel indices [start_vox, end_vox)
                    x_start_vox = int(np.floor(x_start_phys / voxel_size))
                    x_end_vox = int(np.ceil(x_end_phys / voxel_size))
                    y_start_vox = int(np.floor(y_start_phys / voxel_size))
                    y_end_vox = int(np.ceil(y_end_phys / voxel_size))

                    # Clip indices to slice dimensions
                    x_start_vox = max(0, x_start_vox)
                    x_end_vox = min(width, x_end_vox)
                    y_start_vox = max(0, y_start_vox)
                    y_end_vox = min(height, y_end_vox)

                    # Check validity and extract box
                    if x_end_vox > x_start_vox and y_end_vox > y_start_vox:
                        # slice_2d is (height, width) = (Y, X)
                        box = slice_2d[y_start_vox:y_end_vox, x_start_vox:x_end_vox]
                        if np.any(box):  # Check if structure intersects the box
                            count += 1

            current_offset_N[idx] = count
        N_list_sum += current_offset_N

    # Average counts across offsets
    N_list = N_list_sum / num_offsets

    # --- Perform Linear Regression ---
    valid_mask = (N_list > 0)
    L_list_filtered = np.array(L_list)[valid_mask]

    if len(L_list_filtered) < 2: return np.nan

    logL = np.log(L_list_filtered)
    logN = np.log(N_list[valid_mask])

    # Ensure log values are finite
    finite_mask = np.isfinite(logL) & np.isfinite(logN)
    logL_finite = logL[finite_mask]
    logN_finite = logN[finite_mask]

    if len(logL_finite) < 2: return np.nan

    # Linear regression
    slope, _, _, _, _ = stats.linregress(logL_finite, logN_finite)

    # FD = -slope
    return -slope


# --- Main Processing Function ---
def process_single_file(file_path, target_spacing=1.0):
    """
    处理单个NIfTI文件: 加载, 重采样, 二值化, 裁剪, 计算3D FD 和 聚合2D FD 统计量.
    Args:
        file_path (str): Path to the NIfTI file.
        target_spacing (float): Target isotropic resolution in mm.
    Returns:
        tuple: (results_dict, None) on success, or (None, file_path) on failure/skip.
    """
    try:
        # --- Load NIfTI file ---
        img = nib.load(file_path)
        data = img.get_fdata()
        # NIfTI standard shape is (nx, ny, nz), zooms are (dx, dy, dz)
        original_zooms = img.header.get_zooms()[:3]

        # Validate voxel sizes from header
        if not all(z > 0 for z in original_zooms):
            print(f"警告: 文件 {os.path.basename(file_path)} 包含无效的体素尺寸（非正数）。跳过。")
            return None, file_path

        # --- Axis Handling: Convert to Z, Y, X order ---
        # NIfTI data shape is typically (X, Y, Z)
        # Transpose to (Z, Y, X) for consistency with our functions
        data_zyx = np.transpose(data, (2, 1, 0))
        # Corresponding zooms: (dx, dy, dz) -> (dz, dy, dx)
        zooms_zyx = np.array(original_zooms)[::-1]

        # --- Resampling to Isotropic Resolution ---
        # Check if already close to the target resolution to avoid unnecessary resampling
        is_already_isotropic = all(abs(z - target_spacing) < 1e-4 for z in zooms_zyx)

        if not is_already_isotropic:
            # print(f"Resampling {os.path.basename(file_path)} to {target_spacing}mm resolution...")
            resampled_data_zyx = resample_to_isotropic(data_zyx, zooms_zyx, target_spacing)
            voxel_size = target_spacing  # Use target spacing after resampling
        else:
            # print(f"File {os.path.basename(file_path)} is already isotropic. Skipping resampling.")
            resampled_data_zyx = data_zyx
            voxel_size = zooms_zyx[0]  # Use existing isotropic spacing

        # Ensure voxel size is positive after processing
        if voxel_size <= 0:
            print(f"警告: Calculated voxel size is {voxel_size} for file: {os.path.basename(file_path)}. Skipping.")
            return None, file_path

        # --- Binarization ---
        # Convert data to a binary mask (values > 0 become 1, others become 0)
        mask_3d = (resampled_data_zyx > 0).astype(np.uint8)

        # --- Cropping ---
        # Remove empty slices along all axes
        cropped_mask, boundaries = crop_zeros_3d(mask_3d)
        if cropped_mask is None:
            # print(f"Skipping file {os.path.basename(file_path)}: Cropped mask is empty or invalid.")
            return None, file_path  # Mark as skipped

        nz_crop, ny_crop, nx_crop = cropped_mask.shape

        # Check if cropped mask has any volume
        if nz_crop == 0 or ny_crop == 0 or nx_crop == 0:
            # print(f"Skipping file {os.path.basename(file_path)}: Cropped mask has zero volume.")
            return None, file_path

        # --- Calculate 3D FD ---
        fd_3d = calculate_fd_3d(cropped_mask, voxel_size)

        # --- Calculate 2D FDs for each orientation (Axial, Sagittal, Coronal) ---
        # Determine the number of parallel jobs (leave one core free if possible)
        num_cores = multiprocessing.cpu_count()
        n_jobs = max(1, num_cores - 1)

        # 1. Axial Slices (Fixed Z, scan YX plane)
        axial_slices = [cropped_mask[z, :, :] for z in range(nz_crop)]
        if axial_slices:  # Process only if slices exist
            axial_2d_results = Parallel(n_jobs=n_jobs)(
                delayed(calculate_fd_2d)(slice_2d, voxel_size)
                for slice_2d in axial_slices
            )
        else:
            axial_2d_results = []

        # 2. Sagittal Slices (Fixed X, scan ZY plane)
        sagittal_slices = [cropped_mask[:, :, x] for x in range(nx_crop)]
        if sagittal_slices:
            # Transpose ZY slice to YX before passing to calculate_fd_2d
            sagittal_2d_results = Parallel(n_jobs=n_jobs)(
                delayed(calculate_fd_2d)(np.transpose(slice_2d, (1, 0)), voxel_size)
                for slice_2d in sagittal_slices
            )
        else:
            sagittal_2d_results = []

        # 3. Coronal Slices (Fixed Y, scan ZX plane)
        coronal_slices = [cropped_mask[:, y, :] for y in range(ny_crop)]
        if coronal_slices:
            # Transpose ZX slice to YX before passing to calculate_fd_2d
            coronal_2d_results = Parallel(n_jobs=n_jobs)(
                delayed(calculate_fd_2d)(np.transpose(slice_2d, (1, 0)), voxel_size)
                for slice_2d in coronal_slices
            )
        else:
            coronal_2d_results = []

        # --- Aggregate 2D FD results ---
        all_2d_results_lists = [axial_2d_results, sagittal_2d_results, coronal_2d_results]

        # Collect all valid (non-NaN) 2D FD values
        overall_valid_fds = []
        for fd_list in all_2d_results_lists:
            overall_valid_fds.extend([fd for fd in fd_list if not np.isnan(fd)])

        # Calculate summary statistics for the aggregated 2D FDs
        if not overall_valid_fds:
            # If no valid 2D FDs were calculated, set stats to NaN
            overall_max_fd = overall_min_fd = overall_median_fd = overall_mean_fd = np.nan
        else:
            overall_max_fd = np.max(overall_valid_fds)
            overall_min_fd = np.min(overall_valid_fds)
            overall_median_fd = np.median(overall_valid_fds)
            overall_mean_fd = np.mean(overall_valid_fds)

        # --- Prepare results dictionary ---
        results_dict = {
            '文件名': os.path.basename(file_path),
            '文件路径': file_path,
            '3D_FD': fd_3d,
            'Overall_Max_FD': overall_max_fd,
            'Overall_Min_FD': overall_min_fd,
            'Overall_Median_FD': overall_median_fd,
            'Overall_Mean_FD': overall_mean_fd,
        }
        return results_dict, None  # Return results and None for skipped_path (success)

    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return None, file_path
    except nib.spatialimages.ImageFileError:  # Catch errors specifically from nibabel loading
        print(f"错误: 无法加载NIfTI文件 (可能是格式问题或文件损坏): {file_path}")
        return None, file_path
    except Exception as e:
        print(f"处理文件 {file_path} 时发生意外错误: {e}")
        # Uncomment the following lines for detailed error traceback during debugging
        # import traceback
        # traceback.print_exc()
        return None, file_path  # Return None and the path to indicate failure


# --- Directory Processing Function ---
def process_directory(input_dir, output_path, target_spacing=1.0):
    """
    批量处理指定目录中的所有NIfTI文件，并将结果保存到Excel文件.
    Args:
        input_dir (str): Path to the directory containing NIfTI files.
        output_path (str): Path where the output Excel file should be saved.
        target_spacing (float): Target isotropic resolution in mm.
    """
    # --- Collect NIfTI Files ---
    nifti_files = []
    print(f"正在扫描目录: {input_dir}")
    # os.walk efficiently traverses the directory tree
    for root, _, files in os.walk(input_dir):
        for file in files:
            # Check for common NIfTI file extensions (case-insensitive)
            if file.lower().endswith(('.nii', '.nii.gz')):
                nifti_files.append(os.path.join(root, file))

    # Handle case where no NIfTI files are found
    if not nifti_files:
        print(f"在 '{input_dir}' 及其子目录中未找到任何 NIfTI 文件 (.nii, .nii.gz)。")
        return

    print(f"找到 {len(nifti_files)} 个 NIfTI 文件。开始并行处理...")

    results = []
    skipped_files = []  # Keep track of files that failed or were skipped

    # --- Process Files with Progress Bar ---
    # tqdm provides a visual progress indicator
    for file_path in tqdm(nifti_files, desc="处理文件中", unit="file", ncols=70, ascii=True):
        result, skipped_path = process_single_file(file_path, target_spacing)
        if result:
            results.append(result)  # Add successful results
        elif skipped_path:
            skipped_files.append(skipped_path)  # Add path if processing failed/skipped

    print(f"\n并行处理完成。成功处理 {len(results)} 个文件。")

    # --- Save Results ---
    if results:
        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(results)
        try:
            # Create the output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create if path includes a directory
                os.makedirs(output_dir, exist_ok=True)  # exist_ok=True avoids error if dir exists

            # Save DataFrame to an Excel file
            df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"\n结果已成功保存至 Excel 文件: {output_path}")

        except ImportError:
            # Fallback if openpyxl is not installed
            print("\n警告：无法将结果写入 Excel。请确保已安装 'openpyxl' 库 (pip install openpyxl)。")
            print("尝试保存为 CSV 文件...")
            try:
                # Generate CSV path by replacing extension
                csv_path = os.path.splitext(output_path)[0] + ".csv"
                if output_dir:
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                df.to_csv(csv_path, index=False)
                print(f"结果已成功保存为 CSV 文件: {csv_path}")
            except Exception as csv_e:
                print(f"错误：无法保存为 CSV 文件: {csv_e}")
        except Exception as excel_e:
            # Catch other potential errors during Excel writing
            print(f"错误：保存 Excel 文件时发生异常: {excel_e}")
    else:
        # Message if no results were generated
        print("没有生成任何有效结果。")

    # --- Report Skipped/Failed Files ---
    if skipped_files:
        print(f"\n注意：共跳过或处理失败 {len(skipped_files)} 个文件。")
        # Display a sample of skipped files for user feedback
        print("部分失败/跳过的文件示例:")
        # Show up to the first 10 skipped files
        for i, f_path in enumerate(skipped_files[:min(len(skipped_files), 10)]):
            print(f"  - {os.path.basename(f_path)}")
        if len(skipped_files) > 10:
            # Indicate if there are more skipped files than displayed
            print(f"  ... (还有 {len(skipped_files) - 10} 个文件未显示)")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- User Configuration ---
    # !!! IMPORTANT: Update these paths to your specific directories !!!
    # Use raw strings (r"...") or double backslashes ("\\") for Windows paths.
    input_directory = r"D:\3"  # Directory containing the NIfTI files
    output_xlsx_path = r"D:\fractal.xlsx"  # Full path for the output Excel file
    target_resolution = 1.0  # Target isotropic resolution in mm (e.g., 1.0 mm)
    # --- End Configuration ---

    # --- Input Validation ---
    if not os.path.isdir(input_directory):
        print(f"错误：输入目录 '{input_directory}' 不存在或无效。请检查路径设置。")
    else:
        # --- Start Processing ---
        print("-" * 70)
        print("开始执行分形维数批量计算脚本")
        print(f"输入目录: '{input_directory}'")
        print(f"输出文件: '{output_xlsx_path}'")
        print(f"目标各向同性分辨率: {target_resolution} mm")
        print("-" * 70)

        process_directory(input_directory, output_xlsx_path, target_resolution)

        print("\n脚本执行完毕。")