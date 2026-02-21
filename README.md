# Partial Coherence Imaging Simulator

This is a personal hobby project for performing partial coherence analysis on optical systems with a defined Numerical Aperture (NA). It simulates the aerial images typically found in semiconductor exposure tools (lithography scanners) using Abbe's method. I built this while trying out Antigravity for the first time.

If you are curious about exposure devices, optics, or computational lithography, please give it a try!

## Features

- **Interactive UI**: A Tkinter-based GUI for real-time parameter configuration, with Matplotlib visualizations.
- **Customizable Optical Parameters**: Adjust Wavelength ($\lambda$), Lens NA, and Illumination coherence ($\sigma$).
- **Mask Definition**: Simulate Line and Space (L&S) patterns with adjustable width, number of lines, and orientation (Vertical, Horizontal, or Both).
- **Aberrations**: Full support for 36 standard Fringe Zernike Coefficients (in waves) for analyzing aberrations like Coma, Astigmatism, and Spherical.
- **Through-Focus Sweep**: Simulate images through various focus steps to evaluate the depth of focus and contrast curves.
- **Visualizations**: View 1D intensity profiles, 2D aerial images, through-focus contrast curves, and through-focus intensity heatmaps.
- **Data Export**: Easily export the resulting 1D profiles, contrast curves, and heatmaps to CSV format.

## Principle of Partial Coherence Imaging

### Abbe's Formulation
The main simulation engine implements Abbe's theorem by modeling the illumination as an incoherent sum of coherent point sources. The aerial image intensity $I(x,y)$ is computed by integrating the mutually independent coherent images formed by each point source $(s_x, s_y)$ in the continuous illumination pupil limit (e.g., standard disk $\sigma$):

$$ I(x,y) = \int \int_{\text{source}} \left| E_{\text{img}}(x, y ; s_x, s_y) \right|^2 ds_x ds_y $$

For a single given source $(s_x, s_y)$, the incident wave plane tilts across the mask. The mask's diffracted spectrum passes through the objective pupil $P(f_x, f_y)$ to form the coherent image $E_{\text{img}}$:

$$ E_{\text{img}}(x, y; s_x, s_y) = \mathcal{F}^{-1} \left\\{ \mathcal{F} \Big\\{ M(x, y) \cdot e^{i 2\pi (s_x x + s_y y)} \Big\\} \cdot P(f_x, f_y) \right\\} $$

where:
- $M(x, y)$ is the complex transmission function of the photomask.
- $\mathcal{F}$ and $\mathcal{F}^{-1}$ represent the Spatial 2-Dimensional Fourier and Inverse Fourier Transforms respectively.

### Pupil Function & Aberrations

The generalized pupil function is described by the numerical aperture cutoff $\rho \le 1$ and total phase error $\Phi_{\text{total}}$, where $\rho = \frac{\sqrt{f_x^2 + f_y^2}}{\text{NA} / \lambda}$:

$$ P(\rho, \theta) = \Pi(\rho) \cdot \exp( i \Phi_{\text{total}} ) $$

$\Phi_{\text{total}}$ consists of modeled **Wavefront Aberrations** and the exact scalar **Defocus** phase shift. 
Aberrations are injected as a weighted sum of Fringe Zernike polynomials $Z_j(\rho, \theta)$ with user-defined coefficients $c_j$ given in wave variants ($2\pi$ radians):

$$ \Phi_{\text{aberration}}(\rho, \theta) = 2\pi \sum_{j} c_j Z_j(\rho, \theta) $$

Defocus ($z$) propagation natively utilizes the scalar diffraction plane-wave propagation phase. While usually approximated parabolically in low NA, to avoid inaccuracies at high NA, the formula evaluated is the rigorous form:

$$ \Phi_{\text{defocus}}(f_x, f_y; z) = \frac{2\pi}{\lambda} z \sqrt{1 - (\lambda f_x)^2 - (\lambda f_y)^2} $$

## Requirements

The application requires Python 3 and the following dependencies:
- `numpy`
- `matplotlib`

You can install the required packages using pip:
```bash
pip install numpy matplotlib
```

## Usage

Run the main application from the terminal:

```bash
python main.py
```

### Optical Parameters
1. **Wavelength $\lambda$ (nm)**: Source illumination wavelength (e.g., 365.0 for i-line, 193.0 for ArF).
2. **Lens NA**: Numerical Aperture of the objective lens.
3. **Illumination $\sigma$**: Partial coherence factor of the illumination source (0 to 1).
4. **Focus & Sweep**: Set the central focus position and perform a sweep over a defined range and step size.
5. **L&S Width (nm)**: Width of the lines corresponding to your mask pattern.
6. **Precision**: Choose between `Fast (Rough)` for quick explorations or `High (Slow)` for detailed, high-resolution rendering.

### Zernike Coefficients
Expand the bottom left section to input values (in waves) for up to 36 Fringe Zernike aberrations to see their impact on the aerial image and contrast.

### Simulation Modes
- **Run Full Simulation**: Computes the through-focus calculations, providing heatmaps and contrast curves.
- **Run 2D & Profile Only**: Computes only the specific focus position (faster, great for tuning).
- **Export to CSV**: Saves the currently displayed simulation data to a CSV file.

## Acknowledgements
Thanks to the Antigravity system for the development experience.
