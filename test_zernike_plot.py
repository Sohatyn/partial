import numpy as np
import zernike
import matplotlib.pyplot as plt

def plot_zernikes():
    # Grid of rho, theta
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    # Test indices and names
    tests = [
        (4, "Z4 (Defocus)"),
        (5, "Z5 (Astigmatism X)"),
        (6, "Z6 (Astigmatism Y)"),
        (7, "Z7 (Coma X)"),
        (8, "Z8 (Coma Y)"),
        (9, "Z9 (Spherical)")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (idx, name) in enumerate(tests):
        z = zernike.zernike_polynomial(idx, rho, theta)
        # Apply circular mask
        z[rho > 1.0] = np.nan
        
        ax = axes[i]
        im = ax.imshow(z, extent=[-1, 1, -1, 1], origin='lower', cmap='RdBu_r', vmin=-1.5, vmax=1.5)
        ax.set_title(name)
        fig.colorbar(im, ax=ax)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('zernike_test.png')
    print("Saved zernike_test.png")

if __name__ == '__main__':
    plot_zernikes()
