import argparse
import os
import glob

def process_images(base_path):
    pattern = '**/*.png'
    desired_beta = 'beta_0.02'
    retain_after = 'leftImg8bit'

    for img_file in glob.glob(os.path.join(base_path, pattern), recursive=True):
        if desired_beta in img_file:
            # Split the path and filename
            dir_path, filename = os.path.split(img_file)
            # Find the cut point in the filename and adjust to keep up to 'leftImg8bit'
            cut_index = filename.find(retain_after) + len(retain_after)
            new_filename = filename[:cut_index] + '.png'
            # Combine the directory path with the new filename
            new_name = os.path.join(dir_path, new_filename)
            # Check if the target filename already exists
            if os.path.exists(new_name):
                print(f"Skipping rename, target file exists: {new_name}")
                continue
            os.rename(img_file, new_name)
            print(f'Renamed: {filename} to {new_filename}')
        else:
            os.remove(img_file)
            print(f'Deleted: {os.path.basename(img_file)}')

def main():
    parser = argparse.ArgumentParser(description="Process images in a directory.")
    parser.add_argument(
        "--path",
        type=str,
        default="../datasets/foggy_cityscapes/leftImg8bit_foggy",
        help="Path to the dataset directory."
    )

    args = parser.parse_args()
    process_images(args.path)

if __name__ == '__main__':
    main()
