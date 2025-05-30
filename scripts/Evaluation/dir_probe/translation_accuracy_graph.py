import matplotlib.pyplot as plt
import numpy as np

# Data from the image
# Format: [normal, shuffled]
categories = ["Original", "Masked", "No NE"]
original_data = {
    "Original": [60, 45], 
    "Masked": [29, 15],
    "No NE": [32, 17]
}

official_data = {
    "Original": [45, 38],
    "Masked": [11, 5],
    "No NE": [11, 5]
}

unseen_data = {
    "Original": [33, 26],
    "Masked": [5, 3],
    "No NE": [6, 3]
}

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 7), sharey=True)

# Common settings
bar_width = 0.35
# Decrease spacing between bar groups by making the x positions closer
x = np.array([i*0.75 for i in range(len(original_data))])  # Reduce spacing between bar groups by half
ylim = 70  # Maximum y-limit

# Colors - more exact matches to the image
normal_color = '#C6D9F1'  # Light blue color (refined)
shuffled_color = '#8064A2'  # Purple color (refined)

# Function to create bars and add value labels
def create_bars(ax, data, title):
    normal_values = [data[key][0] for key in data.keys()]
    shuffled_values = [data[key][1] for key in data.keys()]
    
    bars1 = ax.bar(x - bar_width/2, normal_values, bar_width, color=normal_color, label='Normal', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + bar_width/2, shuffled_values, bar_width, color=shuffled_color, label='Shuffled', edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, bars in enumerate([(bars1, normal_values), (bars2, shuffled_values)]):
        bar_list, values = bars
        for j, bar in enumerate(bar_list):
            height = values[j]
            # Format with smaller percentage symbol
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height}', ha='center', va='bottom', fontsize=26, fontweight='bold')
            
            # Add smaller percentage symbol with conditional positioning
            # Use different positions for single vs. double digits
            if height < 10:
                # For single digit values
                x_offset = 0.18
            else:
                # For double digit values
                x_offset = 0.25
                
            ax.text(bar.get_x() + bar.get_width()/2. + x_offset, height + 1,
                    '%', ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    # Set title and customize appearance
    ax.set_title(title, fontsize=28, fontweight='bold', pad=20)
    ax.set_xticks(x)
    
    # Match the rotated x-axis labels
    ax.set_xticklabels(categories, fontsize=26, fontweight='bold', rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=24)
    
    # Add horizontal grid lines (lighter and more subtle)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    
    return bars1, bars2

# Create subplots
bars1, bars2 = create_bars(axes[0], original_data, 'Original\nEnglish')
create_bars(axes[1], official_data, 'Official\nTranslations')
create_bars(axes[2], unseen_data, 'Unseen\nTranslations')

# Common y-axis label
fig.text(0.01, 0.5, 'Accuracy (%)', va='center', rotation='vertical', fontsize=22, fontweight='bold')

# Add legend to the right-most subplot (match the exact position and style)
legend = axes[2].legend(fontsize=28, loc='upper right', frameon=True, 
               bbox_to_anchor=(0.95, 0.95))
legend.get_frame().set_linewidth(0.5)
legend.get_frame().set_edgecolor('lightgray')

# Set common y-limit
for ax in axes:
    ax.set_ylim(0, ylim)
    
    # Make y-axis tick labels bold
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
        tick.label1.set_fontsize(24)

# Tight layout
plt.tight_layout(rect=[0.02, 0, 1, 1])
plt.subplots_adjust(wspace=0.025)  # Reduce space between subplots by half (from 0.05 to 0.025)

# Save the figure
plt.savefig('translation_accuracy_graph.pdf', dpi=300, bbox_inches='tight')
plt.show() 