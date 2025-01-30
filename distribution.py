import matplotlib.pyplot as plt
import numpy as np

categories = ['Density1', 'Density2', 'Density3', 'Density4']
benign = [816, 272, 884, 408]
malignant = [2040, 2176, 544, 68]

# Set up positions for bars
x = np.arange(len(categories))
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

rects1 = ax.bar(x - width/2, benign, width, label='Benign', color='skyblue')
rects2 = ax.bar(x + width/2, malignant, width, label='Malignant', color='lightcoral')

ax.set_ylabel('Number of Images')
ax.set_title('Distribution of Classes by Density Level')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

plt.savefig('class_distribution.png')
plt.close()

print("Plot has been saved as 'class_distribution.png'")