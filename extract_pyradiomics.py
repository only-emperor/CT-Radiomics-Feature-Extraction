import os
import csv
from radiomics import featureextractor
# 配置文件路径，通常用于设定如何提取特征的配置文件
# 加载配置文件
extractor = featureextractor.RadiomicsFeatureExtractor('D:\jupyterbook\CT.yaml')
# 图像和掩膜文件夹路径
image_folder = r'D:\66\output_images2'  # 图像文件夹路径
mask_folder = r'D:\66\output_roi2'  # 掩膜文件夹路径
# 获取图像和掩膜文件列表，筛选出 '.nii.gz' 格式的文件
image_files = [f for f in os.listdir(image_folder) if f.endswith('.nii.gz')]
mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.nii.gz')]
# 确保图和掩膜文件数量一致，如果不一致则抛出异常
if len(image_files) != len(mask_files):
    raise ValueError("图像和掩膜文件数量不匹配，请检查文件夹中的文件。")
# 结果保存文件路径
output_csv = 'D:\重复三\瘤内repea3.csv'
# 记录错误文件
error_files = []
# 创建CSV文件并写入表头（假设所有文件的特征键相同）
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    header_written = False  # 标记是否已写入表头
    # 遍历图像文件夹中的每一对图像和掩膜文件，提取特征
    for image_file, mask_file in zip(image_files, mask_files):
        try:
            # 获取图像和掩膜的完整路径
            image_path = os.path.join(image_folder, image_file)
            mask_path = os.path.join(mask_folder, mask_file)
            # 提取特征，打印正在处理的文件信息
            print(f"正在处理文件: {image_file} 和 {mask_file}...")
            result = extractor.execute(image_path, mask_path)
            # 如果是第一次写入，写入表头
            if not header_written:
                header = ['Image_File', 'Mask_File'] + list(result.keys())  # 表头包含图像和掩膜文件名以及特征名
                writer.writerow(header)
                header_written = True
            # 将当前结果保存到CSV文件中
            row = [image_file, mask_file] + list(result.values())
            writer.writerow(row)
            # 输出提取的特征结果
            print(f"提取结果: {image_file} 和 {mask_file}")
            for key, value in result.items():
                print(f'{key}: {value}')
            print("-" * 50)
        except Exception as e:
            error_files.append((image_file, mask_file, str(e)))  # 记录错误文件和错误信息

print(f"所有特征已保存到 {output_csv}")
# 如果有错误文件，输出错误信息
if error_files:
    print("\n以下文件处理失败：")
    for image_file, mask_file, error in error_files:
        print(f"文件: {image_file} 和 {mask_file} 发生错误: {error}")






