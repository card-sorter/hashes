import os
import csv
import cv2
import numpy as np
from PIL import Image
from typing import List  

class ImageHasher:
    def __init__(self):
        pass

    def _generate_phash(self, image: Image.Image) -> bytes:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        
        dct = cv2.dct(np.float32(resized))
        dct_roi = dct[:8, :8]
        
        median = np.median(dct_roi)
        hash_bits = (dct_roi > median).flatten()
        
        hash_bytes = bytearray()
        for i in range(0, len(hash_bits), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(hash_bits) and hash_bits[i + j]:
                    byte_val |= (1 << (7 - j))
            hash_bytes.append(byte_val)
        
        return bytes(hash_bytes)

    def _generate_block_mean_hash(self, image: Image.Image) -> bytes:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        resized = cv2.resize(gray, (16, 16))  
        blocks = []
        
        for i in range(16):
            for j in range(16):
                block_mean = resized[i, j]
                blocks.append(block_mean)
        
        hash_bytes = bytearray()
        for i in range(0, len(blocks), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(blocks):
                    bit_val = 1 if blocks[i + j] > np.mean(blocks) else 0
                    byte_val |= (bit_val << (7 - j))
            hash_bytes.append(byte_val)
        
        return bytes(hash_bytes)

    def _generate_marr_hildreth_hash(self, image: Image.Image) -> bytes:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        
        resized = cv2.resize(laplacian, (24, 24))  
        
        hash_bytes = bytearray()
        flat_laplacian = resized.flatten()
        
        for i in range(0, len(flat_laplacian), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(flat_laplacian):
                    bit_val = 1 if flat_laplacian[i + j] > 0 else 0
                    byte_val |= (bit_val << (7 - j))
            hash_bytes.append(byte_val)
        
        return bytes(hash_bytes)

    def _generate_combined_hash(self, image: Image.Image) -> int:
        phash = self._generate_phash(image)           
        block_hash = self._generate_block_mean_hash(image)  
        marr_hash = self._generate_marr_hildreth_hash(image)  
        
        combined_bytes = phash + block_hash + marr_hash  
        
        combined_int = 0
        for byte in combined_bytes:
            combined_int = (combined_int << 8) | byte
        
        return combined_int

    def _to_bigints(self, input_hash: int) -> List[int]:
        bigints = []
        mask_64bit = (1 << 64) - 1 
        
        for i in range(14):
            chunk = (input_hash >> (64 * (13 - i))) & mask_64bit
            bigints.append(chunk)
        
        return bigints

    def hash_image(self, image_path: str) -> dict:
        """Hash a single image and return results as dictionary"""
        try:
            image = Image.open(image_path)
            
            combined_hash = self._generate_combined_hash(image)
            
            hash_bigints = self._to_bigints(combined_hash)
            
            card_id = os.path.splitext(os.path.basename(image_path))[0]
            
            return {
                'card_id': card_id,
                'combined_hash': combined_hash,
                'hash_bigints': hash_bigints
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

def hash_images_folder(folder_path: str, output_csv: str = "image_hashes.csv"):
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    hasher = ImageHasher()
    
    image_files = []
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported_extensions:
            image_files.append(os.path.join(folder_path, filename))
    
    print(f"Found {len(image_files)} images to process...")
    
    results = []
    for image_path in image_files:
        print(f"Processing: {os.path.basename(image_path)}")
        result = hasher.hash_image(image_path)
        if result is not None:
            results.append(result)
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Card_ID'] + [f'Bigint{i+1}' for i in range(14)]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {'Card_ID': result['card_id']}
            for i, hash_val in enumerate(result['hash_bigints']):
                row[f'Bigint{i+1}'] = hash_val
            writer.writerow(row)
    
    print(f"Successfully processed {len(results)} images")
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    images_folder = "images"
    output_file = "image_hashes.csv"
    
    if not os.path.exists(images_folder):
        print(f"Error: Folder '{images_folder}' does not exist!")
        exit(1)
    
    hash_images_folder(images_folder, output_file)
