import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import csv

# Import our backend
import simulation

ZERNIKE_NAMES = {
    1: "Piston", 2: "X Tilt", 3: "Y Tilt", 4: "Defocus",
    5: "Astigmatism X", 6: "Astigmatism Y", 7: "Coma X", 8: "Coma Y",
    9: "Spherical", 10: "Trefoil X", 11: "Trefoil Y",
    12: "Sec. Astigmatism X", 13: "Sec. Astigmatism Y", 14: "Sec. Coma X", 15: "Sec. Coma Y",
    16: "Sec. Spherical", 17: "Tetrafoil X", 18: "Tetrafoil Y",
    19: "Sec. Trefoil X", 20: "Sec. Trefoil Y", 21: "Ter. Astigmatism X", 22: "Ter. Astigmatism Y",
    23: "Ter. Coma X", 24: "Ter. Coma Y", 25: "Ter. Spherical",
    26: "Pentafoil X", 27: "Pentafoil Y", 28: "Sec. Tetrafoil X", 29: "Sec. Tetrafoil Y",
    30: "Ter. Trefoil X", 31: "Ter. Trefoil Y", 32: "Quat. Astigmatism X", 33: "Quat. Astigmatism Y",
    34: "Quat. Coma X", 35: "Quat. Coma Y", 36: "Quat. Spherical"
}

class PartialCoherenceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Partial Coherence Imaging Simulator")
        self.geometry("1000x800")
        
        # State variables
        self.current_img = None
        self.current_1d = None
        self.current_foc_list = None
        self.current_c_list = None
        self.current_contrast = 0.0
        
        self._build_ui()
        
    def _build_ui(self):
        # Create Main Paned Window
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # --- LEFT PANEL (Inputs) ---
        left_frame = ttk.Frame(main_pane, width=650)
        main_pane.add(left_frame, weight=1)
        
        # Input standard parameters
        param_frame = ttk.LabelFrame(left_frame, text="Optical Parameters")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Wavelength λ (nm):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_wav = tk.StringVar(value="365.0")
        ttk.Entry(param_frame, textvariable=self.var_wav, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Lens NA (0-1):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_na = tk.StringVar(value="0.1")
        ttk.Entry(param_frame, textvariable=self.var_na, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Illumination σ (0-1):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_sig = tk.StringVar(value="0.8")
        ttk.Entry(param_frame, textvariable=self.var_sig, width=10).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(param_frame, text="Focus (um):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_foc = tk.StringVar(value="0.0")
        ttk.Entry(param_frame, textvariable=self.var_foc, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Focus Sweep ±(um):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        span_frame = ttk.Frame(param_frame)
        span_frame.grid(row=4, column=1, sticky=tk.W)
        self.var_foc_span = tk.StringVar(value="5.0")
        ttk.Entry(span_frame, textvariable=self.var_foc_span, width=5).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(span_frame, text="Step:").pack(side=tk.LEFT)
        self.var_foc_step = tk.StringVar(value="0.5")
        ttk.Entry(span_frame, textvariable=self.var_foc_step, width=5).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Label(param_frame, text="L&S Width (nm):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_w = tk.StringVar(value="1800.0")
        ttk.Entry(param_frame, textvariable=self.var_w, width=10).grid(row=5, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Orientation:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        ori_frame = ttk.Frame(param_frame)
        ori_frame.grid(row=6, column=1, sticky=tk.W)
        self.var_ori = tk.StringVar(value="V")
        ttk.Radiobutton(ori_frame, text="Vertical", variable=self.var_ori, value="V").pack(side=tk.LEFT)
        ttk.Radiobutton(ori_frame, text="Horizontal", variable=self.var_ori, value="H").pack(side=tk.LEFT)
        
        # Zernike Parameters (Scrollable)
        z_frame_container = ttk.LabelFrame(left_frame, text="36 Fringe Zernike Coefficients (waves)")
        z_frame_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        canvas = tk.Canvas(z_frame_container)
        scrollbar = ttk.Scrollbar(z_frame_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.zernike_entries = []
        for i in range(1, 37):
            name = ZERNIKE_NAMES.get(i, "")
            row = (i - 1) % 18
            col_base = ((i - 1) // 18) * 2
            
            ttk.Label(scrollable_frame, text=f"Z{i} ({name}):").grid(row=row, column=col_base, sticky=tk.E, padx=5, pady=1)
            e_var = tk.StringVar(value="0.0")
            e = ttk.Entry(scrollable_frame, textvariable=e_var, width=8)
            e.grid(row=row, column=col_base+1, padx=5, pady=1)
            self.zernike_entries.append(e_var)
            
        # Action Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        ttk.Button(btn_frame, text="Run Simulation", command=self.run_simulation).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Export to CSV", command=self.export_csv).pack(fill=tk.X, pady=2)
        
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(left_frame, textvariable=self.status_var, foreground="blue").pack(anchor=tk.W, padx=5)
        self.contrast_lbl = tk.StringVar(value="Center Contrast: N/A")
        ttk.Label(left_frame, textvariable=self.contrast_lbl, font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        
        # --- RIGHT PANEL (Outputs) ---
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=3)
        
        # Matplotlib Figures
        self.fig = Figure(figsize=(7, 10), dpi=100)
        self.ax1 = self.fig.add_subplot(311)  # 1D Profile
        self.ax2 = self.fig.add_subplot(312)  # Contrast Curve
        self.ax3 = self.fig.add_subplot(313)  # Heatmap
        self.fig.tight_layout(pad=4.0)
        
        # Colorbar reference for heatmap
        self.cbar = None
        self.cbar_ax = None
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _get_inputs(self):
        try:
            wav = float(self.var_wav.get())
            na = float(self.var_na.get())
            sig = float(self.var_sig.get())
            foc_um = float(self.var_foc.get())
            foc_span = float(self.var_foc_span.get())
            foc_step = float(self.var_foc_step.get())
            w = float(self.var_w.get())
            ori = self.var_ori.get()
            z_coeffs = np.array([float(v.get()) for v in self.zernike_entries])
            return wav, na, sig, foc_um, foc_span, foc_step, w, ori, z_coeffs
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all inputs are valid numbers.")
            return None

    def run_simulation(self):
        params = self._get_inputs()
        if not params: return
        wav, na, sig, foc_um, foc_span, foc_step, w, ori, z_coeffs = params
        
        self.status_var.set("Running simulation... please wait.")
        self.update()
        
        try:
            # Resolution logic
            # Ensure the image array covers at least 10 times the line width
            target_field_size = 10.0 * w
            Nx, Ny = 512, 512
            pixel_size = target_field_size / Nx
            
            # --- 1. Single Focus Simulation ---
            foc_nm = foc_um * 1000.0
            mask = simulation.generate_mask(Nx, Ny, pixel_size, w, ori)
            src = simulation.get_source_points(na, sig, wav, num_points=120)
            
            img = simulation.simulate_image(mask, na, sig, wav, foc_nm, z_coeffs, pixel_size, source_points=src)
            c = simulation.calculate_contrast(img, w, pixel_size, ori)
            
            self.current_img = img
            self.current_contrast = c
            self.contrast_lbl.set(f"Center Contrast: {c:.4f}")
            
            # Extract 1D Center Profile
            cx, cy = Nx//2, Ny//2
            if ori == 'V':
                profile = img[cy, :]
                x_axis = (np.arange(Nx) - cx) * pixel_size
            else:
                profile = img[:, cx]
                x_axis = (np.arange(Ny) - cy) * pixel_size
            self.current_1d = (x_axis, profile)
            
            # --- 2. Through-Focus Sweep ---
            span_um = foc_span
            step_um = foc_step
            if step_um <= 0:
                raise ValueError("Focus step must be > 0.")
            num_steps = int(round(2 * span_um / step_um)) + 1
            foc_um_list = np.linspace(foc_um - span_um, foc_um - span_um + (num_steps - 1) * step_um, num_steps)
            foc_nm_list = foc_um_list * 1000.0
            
            c_list, p_list = simulation.run_through_focus(
                w, na, sig, wav, foc_nm_list, z_coeffs, ori, Nx, Ny, pixel_size, num_source=50
            ) # fewer points to save time
            
            self.current_foc_list = foc_um_list
            self.current_c_list = c_list
            
            # Plot
            self._update_plots(x_axis, profile, foc_um, c, foc_um_list, c_list, p_list, w)
            self.status_var.set("Simulation completed.")
            
        except Exception as e:
            messagebox.showerror("Simulation Error", str(e))
            self.status_var.set("Error occurred.")
            
    def _update_plots(self, x, prof, f_user, c_user, f_list, c_list, p_list, w):
        # 1. 1D Profile
        self.ax1.clear()
        self.ax1.plot(x, prof, 'b-', label='Aerial Image')
        self.ax1.set_title(f"Image Profile at Focus = {f_user:.3f} um")
        self.ax1.set_xlabel("Position (nm)")
        self.ax1.set_ylabel("Intensity (a.u.)")
        self.ax1.grid(True)
        self.ax1.set_xlim([-4.5 * w, 4.5 * w])
        self.ax1.legend()
        
        # 2. Contrast Curve
        self.ax2.clear()
        self.ax2.plot(f_list, c_list, 'k-o', label='Contrast Curve')
        self.ax2.plot(f_user, c_user, 'r*', markersize=12, label='Current Focus')
        self.ax2.set_title("Through-Focus Contrast")
        self.ax2.set_xlabel("Focus (um)")
        self.ax2.set_ylabel("Contrast")
        self.ax2.grid(True)
        self.ax2.legend()
        
        # 3. Heatmap
        self.ax3.clear()
        # p_list is shape (num_focus, num_x)
        # To plot x vs focus, we use imshow or pcolormesh
        # X axis: physical position
        # Y axis: focus
        
        # Create custom colormap: Green (darkest) to Red (brightest)
        # Or standard RdYlGn reversed so green is low and red is high
        cmap = matplotlib.colormaps['RdYlGn_r']
        
        # For pcolormesh, we need coordinate edges
        X_mesh, Y_mesh = np.meshgrid(x, f_list)
        
        pcm = self.ax3.pcolormesh(X_mesh, Y_mesh, p_list, cmap=cmap, shading='auto')
        self.ax3.set_title("Through-Focus Intensity Heatmap")
        self.ax3.set_xlabel("Position (nm)")
        self.ax3.set_ylabel("Focus (um)")
        self.ax3.set_xlim([-4.5 * w, 4.5 * w])
        
        if self.cbar_ax is not None:
            self.cbar_ax.remove()
        
        # Add custom axis for colorbar to prevent layout issues on refresh
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(self.ax3)
        self.cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
        self.cbar = self.fig.colorbar(pcm, cax=self.cbar_ax, orientation='vertical', label='Intensity')
        
        self.fig.tight_layout(pad=3.0)
        self.canvas_plot.draw()
        
    def export_csv(self):
        if self.current_1d is None or self.current_foc_list is None:
            messagebox.showwarning("No Data", "Please run a simulation first!")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="Export Simulation Data"
        )
        if not filepath:
            return
            
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["--- 1D Image Profile ---"])
                writer.writerow(["Position (nm)", "Intensity (a.u.)"])
                x, prof = self.current_1d
                for xx, pp in zip(x, prof):
                    writer.writerow([xx, pp])
                    
                writer.writerow([])
                writer.writerow(["--- Through-Focus Contrast ---"])
                writer.writerow(["Focus (um)", "Contrast"])
                for ff, cc in zip(self.current_foc_list, self.current_c_list):
                    writer.writerow([ff, cc])
                    
            self.status_var.set(f"Successfully exported data to CSV.")
            messagebox.showinfo("Export Successful", f"Data saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save CSV:\n{str(e)}")

if __name__ == "__main__":
    app = PartialCoherenceApp()
    app.mainloop()
