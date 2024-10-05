import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot the model performance (in terms of accuracy) from CSV with a vertical line indicating the epoch when the watermark was injected.')
parser.add_argument('file_path', type=str, help='complete path to the CSV file such as train.csv or val.csv located in the outputs folder. ')
parser.add_argument('epoch_line', type=int, help='Epoch number when the watermarking took place. E.g., 58')

args = parser.parse_args()
file_path = args.file_path
epoch_line = args.epoch_line

data = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))

# this will be the vertical line indicating the epoch when the model was watermarked. This value needs to be passed as an argument. The value is located on teh tattooed.log of the individual run
plt.axvline(x=epoch_line, color='#fcb404', linestyle='-')

# Plot accuracy with a red line
plt.plot(data['epoch'], data['accuracy'], linestyle='-', color='#219ebc', label='Accuracy')

# Add grid lines
plt.grid(True)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
#  use savefig functionality if you want to store the plot as a pdf
plt.show()

