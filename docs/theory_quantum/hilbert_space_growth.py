import matplotlib.pyplot as plt
import numpy as np

def generate_hilbert_space_plot():
    """
    Generates and saves a plot comparing the growth of classical and quantum state spaces.
    """
    # --- Data Preparation ---
    N = np.arange(1, 15)  # Number of particles/qubits
    # Classical system size grows linearly (e.g., 2 states per particle)
    classical_dim = 2 * N
    # Quantum system size grows exponentially (2^N complex numbers for N qubits)
    quantum_dim = 2**N

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(N, classical_dim, 'o-', color='cornflowerblue', label='Classical State Space (Size = 2N)', lw=2)
    ax.plot(N, quantum_dim, 's-', color='indianred', label=r'Quantum Hilbert Space (Dimension = $2^N$)', lw=2)

    # Using a logarithmic scale for the y-axis to make the quantum growth visible
    ax.set_yscale('log')

    # --- Labels and Titles ---
    ax.set_xlabel('Number of Qubits (N)', fontsize=14)
    ax.set_ylabel('Dimension of State Space (log scale)', fontsize=14)
    title = 'The "Curse of Dimensionality": Quantum vs. Classical'
    ax.set_title(title, fontsize=16, pad=20)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()

    # --- Save the figure ---
    filename = "hilbert_space_growth.png"
    plt.savefig(filename)
    print(f"Image saved as {filename}")
    plt.show()

if __name__ == '__main__':
    generate_hilbert_space_plot()
