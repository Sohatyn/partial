import simulation
import numpy as np
import matplotlib.pyplot as plt

def test_coma():
    Nx, Ny = 512, 512
    pixel_size = 2.0
    lw = 40.0
    wav = 193.0
    NA = 1.35
    sig = 0.8
    foc = 0.0
    
    # Aberration: Coma X (Z7) = 0.1 waves
    z_coeffs_coma = np.zeros(36)
    z_coeffs_coma[6] = 0.15 # Z7 -> index 6
    
    mask = simulation.generate_mask(Nx, Ny, pixel_size, lw, 'V')
    src = simulation.get_source_points(NA, sig, wav, num_points=100)
    
    img = simulation.simulate_image(mask, NA, sig, wav, foc, z_coeffs_coma, pixel_size, source_points=src)
    
    # V line should show asymmetry in X direction due to Coma X
    cx, cy = Nx//2, Ny//2
    profile = img[cy, cx-40:cx+41] # +/- 80nm roughly
    x_axis = (np.arange(len(profile)) - 40) * pixel_size
    
    # Left vs Right peak intensity comparison
    left_side = profile[:40]
    right_side = profile[41:]
    
    print("--- Coma X Test (Vertical Lines) ---")
    print(f"Max intensity on Left: {np.max(left_side):.4f}")
    print(f"Max intensity on Right: {np.max(right_side):.4f}")
    print(f"Difference (L - R): {np.max(left_side) - np.max(right_side):.4f}")
    
    plt.figure()
    plt.plot(x_axis, profile, label="Coma X = 0.15 waves")
    plt.title("Image Profile with Coma X")
    plt.xlabel("Position (nm)")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.savefig('coma_test.png')
    print("Saved coma_test.png")

if __name__ == '__main__':
    test_coma()
