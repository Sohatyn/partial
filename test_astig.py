import simulation
import numpy as np

def test_astigmatism():
    # Test parameters
    Nx, Ny = 512, 512
    pixel_size = 2.0
    lw = 40.0
    wav = 193.0
    NA = 1.35
    sig = 0.8
    foc_list = np.linspace(-150, 150, 31)
    
    # Base case: No aberration
    z_coeffs_base = np.zeros(36)
    
    # Aberration: Astigmatism X (Z5) = 0.1 waves
    z_coeffs_astig = np.zeros(36)
    z_coeffs_astig[4] = 0.1  # Z5 -> index 4
    
    print("Running base case (V and H)...")
    base_c_v, _ = simulation.run_through_focus(lw, NA, sig, wav, foc_list, z_coeffs_base, 'V', Nx, Ny, pixel_size, num_source=50)
    base_c_h, _ = simulation.run_through_focus(lw, NA, sig, wav, foc_list, z_coeffs_base, 'H', Nx, Ny, pixel_size, num_source=50)
    
    print("Running Astigmatism case (V and H)...")
    astig_c_v, _ = simulation.run_through_focus(lw, NA, sig, wav, foc_list, z_coeffs_astig, 'V', Nx, Ny, pixel_size, num_source=50)
    astig_c_h, _ = simulation.run_through_focus(lw, NA, sig, wav, foc_list, z_coeffs_astig, 'H', Nx, Ny, pixel_size, num_source=50)
    
    # Find best focus (max contrast index)
    bf_base_v = foc_list[np.argmax(base_c_v)]
    bf_base_h = foc_list[np.argmax(base_c_h)]
    
    bf_astig_v = foc_list[np.argmax(astig_c_v)]
    bf_astig_h = foc_list[np.argmax(astig_c_h)]
    
    print("\n--- Results ---")
    print(f"Base Best Focus (V): {bf_base_v:.1f} nm")
    print(f"Base Best Focus (H): {bf_base_h:.1f} nm")
    print(f"Base Astigmatism difference (V-H): {bf_base_v - bf_base_h:.1f} nm")
    
    print(f"\nAstigmatism Best Focus (V): {bf_astig_v:.1f} nm")
    print(f"Astigmatism Best Focus (H): {bf_astig_h:.1f} nm")
    print(f"Astigmatism difference (V-H): {bf_astig_v - bf_astig_h:.1f} nm")

if __name__ == '__main__':
    test_astigmatism()
