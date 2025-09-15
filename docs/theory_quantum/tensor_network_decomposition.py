import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def generate_tensor_decomposition_diagram():
    """
    Generates and saves a detailed diagram showing the decomposition of a large
    tensor into a structured tensor network with clear annotations.
    """
    # --- Setup ---
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_aspect('equal')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # --- Left Side: Large Complex Tensor ---
    large_tensor = patches.Circle((2.5, 3.5), radius=1.0, facecolor='indianred', edgecolor='black', lw=1.5)
    ax.add_patch(large_tensor)
    ax.text(2.5, 3.5, r'$T_{i_1 i_2 ... i_{16}}$', ha='center', va='center', fontsize=14, color='white')
    ax.text(2.5, 1.8, '(Computationally Intractable)', ha='center', va='center', fontsize=12)


    # Physical indices for the large tensor
    for i in range(16):
        angle = (i / 16) * 2 * np.pi
        start_x = 2.5 + 1.0 * np.cos(angle)
        start_y = 3.5 + 1.0 * np.sin(angle)
        end_x = 2.5 + 1.5 * np.cos(angle)
        end_y = 3.5 + 1.5 * np.sin(angle)
        ax.plot([start_x, end_x], [start_y, end_y], color='black', lw=1.5)

    # --- Arrow ---
    ax.arrow(4.8, 3.5, 1.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black', lw=2)
    ax.text(5.5, 3.8, 'Decomposition', ha='center', fontsize=12)

    # --- Right Side: Tensor Network ---
    ax.text(9.5, 1.8, '(Efficient Representation)', ha='center', va='center', fontsize=12)
    # Create a 4x4 grid of smaller tensors
    for i in range(4):
        for j in range(4):
            pos = (8 + i, 5.5 - j)
            tensor_node = patches.Circle(pos, radius=0.2, facecolor='cornflowerblue', edgecolor='black', lw=1)
            ax.add_patch(tensor_node)
            ax.text(pos[0], pos[1], f'$t_{j*4+i}$', ha='center', va='center', fontsize=8, color='white')

            # Horizontal connections (contracted indices)
            if i > 0:
                ax.plot([pos[0] - 0.8, pos[0] - 0.2], [pos[1], pos[1]], color='gray', lw=2)
            # Vertical connections (contracted indices)
            if j > 0:
                ax.plot([pos[0], pos[0]], [pos[1] + 0.2, pos[1] + 0.8], color='gray', lw=2)
            # Physical (open) indices
            if i == 0: ax.plot([pos[0] - 0.2, pos[0] - 0.5], [pos[1], pos[1]], color='black', lw=1.5)
            if i == 3: ax.plot([pos[0] + 0.2, pos[0] + 0.5], [pos[1], pos[1]], color='black', lw=1.5)
            if j == 0: ax.plot([pos[0], pos[0]], [pos[1] + 0.2, pos[1] + 0.5], color='black', lw=1.5)
            if j == 3: ax.plot([pos[0], pos[0]], [pos[1] - 0.2, pos[1] - 0.5], color='black', lw=1.5)

    # --- Add Annotations for Clarity ---
    # Annotation for Physical Index
    ax.annotate('Physical Index\n(e.g., a qubit)',
                xy=(11.5, 6.0), xycoords='data',
                xytext=(12.5, 6.5), textcoords='data',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", lw=1.5),
                fontsize=12, ha='center')

    # Annotation for Contracted Bond Index
    ax.annotate('Contracted Bond Index\n(Represents Entanglement)',
                xy=(9.5, 4.0), xycoords='data',
                xytext=(11.5, 3.0), textcoords='data',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='gray', lw=1.5),
                fontsize=12, ha='center')

    title = 'Tensor Network as a Structured Representation of a Quantum State'
    fig.suptitle(title, fontsize=16)

    # --- Save the figure ---
    filename = "tensor_network_decomposition.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    print(f"Image saved as {filename}")
    plt.show()

if __name__ == '__main__':
    generate_tensor_decomposition_diagram()

