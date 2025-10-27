import os
import csv

# Define the folder path
folder_path = r'D:\Projects\RM_Dataset\Indus Dataset'

# Open a CSV file to write
with open('folder_contents.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write headers (you can modify this as per your need)
    writer.writerow(['Filename', 'File Size (bytes)', 'Creation Time', 'Last Modified Time'])
    
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's a file (and not a subdirectory)
        if os.path.isfile(file_path):
            # Gather file details
            file_size = os.path.getsize(file_path)
            creation_time = os.path.getctime(file_path)
            modified_time = os.path.getmtime(file_path)
            
            # Write row to CSV
            writer.writerow([filename, file_size, creation_time, modified_time])

print("Folder contents have been written to folder_contents.csv.")
