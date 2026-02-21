import numpy as np
import zernike

def test_zernike():
    print("--- Verifying Fringe Zernike Polynomials ---")
    
    # Grid of rho, theta
    rho = np.linspace(0, 1, 100)
    theta = np.zeros_like(rho) # testing theta=0
    
    # 1. Test Piston (Z1) -> R(0,0) = 1
    z1 = zernike.zernike_polynomial(1, rho, theta)
    print(f"Z1 (Piston) values: min={np.min(z1):.4f}, max={np.max(z1):.4f}")
    assert np.allclose(z1, 1.0), "Z1 should be all 1s"
    
    # 2. Test Defocus (Z4) -> R(2,0) = 2*rho^2 - 1
    z4 = zernike.zernike_polynomial(4, rho, theta)
    expected_z4 = 2 * rho**2 - 1
    print(f"Z4 (Defocus) max diff from expected: {np.max(np.abs(z4 - expected_z4)):.4e}")
    assert np.allclose(z4, expected_z4), "Z4 does not match expected formula"
    
    # 3. Test Spherical (Z9) -> R(4,0) = 6*rho^4 - 6*rho^2 + 1
    z9 = zernike.zernike_polynomial(9, rho, theta)
    expected_z9 = 6 * rho**4 - 6 * rho**2 + 1
    print(f"Z9 (Spherical) max diff from expected: {np.max(np.abs(z9 - expected_z9)):.4e}")
    assert np.allclose(z9, expected_z9), "Z9 does not match expected formula"
    
    # 4. Test Coma X (Z7) -> R(3,1)*cos(t) = (3*rho^3 - 2*rho) * cos(t)
    z7_theta0 = zernike.zernike_polynomial(7, rho, theta)
    expected_z7 = 3 * rho**3 - 2 * rho # cos(0)=1
    print(f"Z7 (Coma X) at theta=0 max diff: {np.max(np.abs(z7_theta0 - expected_z7)):.4e}")
    
    z7_theta_pi_2 = zernike.zernike_polynomial(7, rho, np.ones_like(rho) * np.pi/2)
    print(f"Z7 (Coma X) at theta=pi/2 max abs value: {np.max(np.abs(z7_theta_pi_2)):.4e}") # should be 0 because cos(pi/2)=0
    
    # 5. Check mapping dictionary lengths/keys
    print(f"Total defined Fringe polynomials: {len(zernike.FRINGE_36)}")
    
    print("--- Zernike Test Passed! ---")

if __name__ == '__main__':
    test_zernike()
