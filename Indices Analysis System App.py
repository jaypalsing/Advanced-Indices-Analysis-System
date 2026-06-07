# ============================================================
# ADVANCED REMOTE SENSING & GIS ANALYSIS SYSTEM
# ============================================================

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import rasterio

from rasterio.enums import Resampling
from rasterio.warp import reproject

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import seaborn as sns


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def safe_divide(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(num, den)
        result[~np.isfinite(result)] = np.nan
    return result


def read_raster(path):
    src = rasterio.open(path)
    arr = src.read(1).astype("float32")

    if src.nodata is not None:
        arr = np.where(arr == src.nodata, np.nan, arr)

    return arr, src


def same_grid(src, ref):
    return (
        src.width == ref.width and
        src.height == ref.height and
        src.transform == ref.transform and
        src.crs == ref.crs
    )


def reproject_to_reference(src, ref):
    output = np.empty(
        (ref.height, ref.width),
        dtype="float32"
    )

    reproject(
        source=rasterio.band(src, 1),
        destination=output,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ref.transform,
        dst_crs=ref.crs,
        resampling=Resampling.bilinear
    )

    return output


def percent_clip(arr, pmin=2, pmax=98):
    data = arr[~np.isnan(arr)]

    if data.size == 0:
        return None, None

    return (
        np.percentile(data, pmin),
        np.percentile(data, pmax)
    )


def calculate_vci(ndvi):
    ndvi_min = np.nanmin(ndvi)
    ndvi_max = np.nanmax(ndvi)

    return safe_divide(
        ndvi - ndvi_min,
        ndvi_max - ndvi_min
    ) * 100


# ============================================================
# INDEX FORMULAS
# ============================================================

INDEX_SPECS = {

    "NDVI": {
        "needs": ["NIR", "RED"],
        "formula": lambda NIR, RED, **_: safe_divide(
            NIR - RED,
            NIR + RED
        )
    },

    "EVI": {
        "needs": ["NIR", "RED", "BLUE"],
        "formula": lambda NIR, RED, BLUE, **_: 2.5 * safe_divide(
            NIR - RED,
            NIR + 6 * RED - 7.5 * BLUE + 1
        )
    },

    "SAVI": {
        "needs": ["NIR", "RED"],
        "formula": lambda NIR, RED, **_: 1.5 * safe_divide(
            NIR - RED,
            NIR + RED + 0.5
        )
    },

    "NDWI": {
        "needs": ["GREEN", "NIR"],
        "formula": lambda GREEN, NIR, **_: safe_divide(
            GREEN - NIR,
            GREEN + NIR
        )
    },

    "MNDWI": {
        "needs": ["GREEN", "SWIR"],
        "formula": lambda GREEN, SWIR, **_: safe_divide(
            GREEN - SWIR,
            GREEN + SWIR
        )
    },

    "NDBI": {
        "needs": ["SWIR", "NIR"],
        "formula": lambda SWIR, NIR, **_: safe_divide(
            SWIR - NIR,
            SWIR + NIR
        )
    },

    "GNDVI": {
        "needs": ["NIR", "GREEN"],
        "formula": lambda NIR, GREEN, **_: safe_divide(
            NIR - GREEN,
            NIR + GREEN
        )
    },

    "MSAVI": {
        "needs": ["NIR", "RED"],
        "formula": lambda NIR, RED, **_: (
            2 * NIR + 1 -
            np.sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - RED))
        ) / 2
    },

    "NBR": {
        "needs": ["NIR", "SWIR"],
        "formula": lambda NIR, SWIR, **_: safe_divide(
            NIR - SWIR,
            NIR + SWIR
        )
    },

    "VCI": {
        "needs": ["NIR", "RED"],
        "formula": lambda NIR, RED, **_: calculate_vci(
            safe_divide(NIR - RED, NIR + RED)
        )
    }
}


# ============================================================
# MAIN APPLICATION
# ============================================================

class RemoteSensingApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Advanced Remote Sensing & GIS Analysis System")
        self.geometry("1800x1000")

        self.loaded_files = []
        self.band_paths = {}
        self.reference_src = None
        self.reference_profile = None
        self.last_result = None
        self.last_title = None

        self.build_ui()

    # ========================================================
    # UI DESIGN
    # ========================================================

    def build_ui(self):

        title = tk.Label(
            self,
            text="Advanced Remote Sensing & GIS Analysis System",
            font=("Arial", 24, "bold"),
            bg="#cfe8ff",
            pady=10
        )
        title.pack(fill=tk.X)

        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(
            main_frame,
            width=320,
            bg="white"
        )
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.right_frame = tk.Frame(
            main_frame,
            bg="#f2f2f2"
        )
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        load_btn = tk.Button(
            left_frame,
            text="Load Raster Folder",
            bg="#ffb74d",
            font=("Arial", 12, "bold"),
            command=self.load_folder
        )
        load_btn.pack(fill=tk.X, padx=5, pady=5)

        self.listbox = tk.Listbox(
            left_frame,
            height=12
        )
        self.listbox.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

        tk.Label(
            left_frame,
            text="Detected Bands",
            bg="white",
            font=("Arial", 13, "bold")
        ).pack(pady=5)

        self.band_text = tk.Text(
            left_frame,
            height=8,
            font=("Consolas", 9)
        )
        self.band_text.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(
            left_frame,
            text="Compute Indices",
            bg="white",
            font=("Arial", 14, "bold")
        ).pack(pady=5)

        for idx_name in INDEX_SPECS.keys():
            tk.Button(
                left_frame,
                text=idx_name,
                bg="#90caf9",
                font=("Arial", 10),
                command=lambda n=idx_name: self.compute_index(n)
            ).pack(fill=tk.X, padx=5, pady=2)

        tk.Button(
            left_frame,
            text="Save Current GeoTIFF",
            bg="#a5d6a7",
            font=("Arial", 11, "bold"),
            command=self.save_current_geotiff
        ).pack(fill=tk.X, padx=5, pady=10)

        tk.Button(
            left_frame,
            text="Clear Dashboard",
            bg="#ef9a9a",
            font=("Arial", 11, "bold"),
            command=self.clear_dashboard
        ).pack(fill=tk.X, padx=5, pady=5)

        self.status = tk.Label(
            self,
            text="Ready",
            relief=tk.SUNKEN,
            anchor="w"
        )
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # ========================================================
    # LOAD FOLDER
    # ========================================================

    def load_folder(self):

        folder = filedialog.askdirectory()

        if not folder:
            return

        self.loaded_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".tif", ".tiff"))
        ]

        if not self.loaded_files:
            messagebox.showwarning(
                "No Raster Files",
                "No .tif or .tiff files found."
            )
            return

        self.listbox.delete(0, tk.END)

        for file in self.loaded_files:
            self.listbox.insert(tk.END, file)

        if self.reference_src:
            self.reference_src.close()

        self.reference_src = rasterio.open(self.loaded_files[0])
        self.reference_profile = self.reference_src.profile.copy()

        self.auto_detect_bands()

        self.status.config(
            text=f"Loaded {len(self.loaded_files)} raster files"
        )

        messagebox.showinfo(
            "Success",
            f"Loaded {len(self.loaded_files)} raster files successfully."
        )

    # ========================================================
    # AUTO BAND DETECTION
    # ========================================================

    def auto_detect_bands(self):

        self.band_paths = {}

        for file in self.loaded_files:

            name = os.path.basename(file).lower()

            # Landsat / Sentinel common names
            if "b2" in name or "band2" in name or "blue" in name or "b02" in name:
                self.band_paths["BLUE"] = file

            elif "b3" in name or "band3" in name or "green" in name or "b03" in name:
                self.band_paths["GREEN"] = file

            elif "b4" in name or "band4" in name or "red" in name or "b04" in name:
                self.band_paths["RED"] = file

            elif "b5" in name or "band5" in name or "nir" in name or "b08" in name:
                self.band_paths["NIR"] = file

            elif (
                "b6" in name or
                "b7" in name or
                "band6" in name or
                "band7" in name or
                "swir" in name or
                "b11" in name or
                "b12" in name
            ):
                self.band_paths["SWIR"] = file

        self.band_text.delete("1.0", tk.END)

        for role, path in self.band_paths.items():
            self.band_text.insert(
                tk.END,
                f"{role}: {os.path.basename(path)}\n"
            )

    # ========================================================
    # LOAD BAND
    # ========================================================

    def load_band(self, role):

        path = self.band_paths.get(role)

        if path is None:
            messagebox.showerror(
                "Missing Band",
                f"{role} band not found.\nPlease check file names."
            )
            return None

        arr, src = read_raster(path)

        try:
            if not same_grid(src, self.reference_src):
                arr = reproject_to_reference(src, self.reference_src)
        finally:
            src.close()

        return arr

    # ========================================================
    # COMPUTE INDEX
    # ========================================================

    def compute_index(self, index_name):

        if self.reference_src is None:
            messagebox.showwarning(
                "No Data",
                "Please load raster folder first."
            )
            return

        spec = INDEX_SPECS[index_name]
        arrays = {}

        for role in spec["needs"]:

            arr = self.load_band(role)

            if arr is None:
                return

            arrays[role] = arr

        try:
            result = spec["formula"](**arrays).astype("float32")
        except Exception as e:
            messagebox.showerror(
                "Calculation Error",
                str(e)
            )
            return

        self.last_result = result
        self.last_title = index_name

        self.display_dashboard(result, index_name)

    # ========================================================
    # DISPLAY ADVANCED DASHBOARD
    # ========================================================

    def display_dashboard(self, arr, title):

        self.clear_dashboard()

        valid_data = arr[~np.isnan(arr)]

        if valid_data.size == 0:
            messagebox.showwarning(
                "No Valid Data",
                "No valid pixel values found."
            )
            return

        full_valid_count = valid_data.size

        if valid_data.size > 50000:
            data = np.random.choice(
                valid_data,
                50000,
                replace=False
            )
        else:
            data = valid_data

        vmin, vmax = percent_clip(arr)

        fig = plt.figure(figsize=(18, 12))

        fig.suptitle(
            f"{title} Advanced Analytics Dashboard",
            fontsize=22,
            fontweight="bold"
        )

        # Main raster map
        ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)

        img = ax1.imshow(
            arr,
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax
        )

        ax1.set_title(f"{title} Raster Map", fontsize=14)
        ax1.axis("off")

        plt.colorbar(
            img,
            ax=ax1,
            fraction=0.046,
            pad=0.04
        )

        # Histogram + KDE
        ax2 = plt.subplot2grid((3, 4), (0, 2))

        sns.histplot(
            data,
            bins=50,
            kde=True,
            color="green",
            ax=ax2
        )

        ax2.set_title("Histogram + KDE")
        ax2.set_xlabel("Pixel Value")
        ax2.set_ylabel("Frequency")

        # Boxplot
        ax3 = plt.subplot2grid((3, 4), (0, 3))

        sns.boxplot(
            y=data,
            color="orange",
            ax=ax3
        )

        ax3.set_title("Boxplot")
        ax3.set_ylabel("Pixel Value")

        # Violin plot
        ax4 = plt.subplot2grid((3, 4), (1, 2))

        sns.violinplot(
            y=data,
            color="skyblue",
            ax=ax4
        )

        ax4.set_title("Violin Plot")
        ax4.set_ylabel("Pixel Value")

        # Scatter plot
        ax5 = plt.subplot2grid((3, 4), (1, 3))

        sample_size = min(10000, len(data))

        sample = np.random.choice(
            data,
            sample_size,
            replace=False
        )

        ax5.scatter(
            np.arange(sample_size),
            sample,
            s=2,
            alpha=0.5,
            color="red"
        )

        ax5.set_title("Pixel Scatter Plot")
        ax5.set_xlabel("Pixel Index")
        ax5.set_ylabel("Value")

        # Line profile
        ax6 = plt.subplot2grid((3, 4), (2, 0))

        line_profile = arr[arr.shape[0] // 2, :]

        ax6.plot(
            line_profile,
            color="blue",
            linewidth=1
        )

        ax6.set_title("Middle Row Line Profile")
        ax6.set_xlabel("Column")
        ax6.set_ylabel("Value")

        # Pie chart
        ax7 = plt.subplot2grid((3, 4), (2, 1))

        very_low = np.sum(data < 0.2)
        low = np.sum((data >= 0.2) & (data < 0.4))
        moderate = np.sum((data >= 0.4) & (data < 0.6))
        high = np.sum(data >= 0.6)

        pie_values = [very_low, low, moderate, high]
        pie_labels = ["Very Low", "Low", "Moderate", "High"]

        ax7.pie(
            pie_values,
            labels=pie_labels,
            autopct="%1.1f%%"
        )

        ax7.set_title("Class Distribution")

        # Heatmap
        ax8 = plt.subplot2grid((3, 4), (2, 2))

        small_arr = arr[::10, ::10]

        sns.heatmap(
            small_arr,
            cmap="RdYlGn",
            cbar=False,
            ax=ax8
        )

        ax8.set_title("Spatial Heatmap")
        ax8.axis("off")

        # Correlation matrix
        ax9 = plt.subplot2grid((3, 4), (2, 3))

        df = pd.DataFrame({
            "Value": data,
            "Square": data ** 2,
            "Sqrt": np.sqrt(np.abs(data)),
            "Log": np.log1p(np.abs(data))
        })

        corr = df.corr()

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            ax=ax9
        )

        ax9.set_title("Correlation Matrix")

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            master=self.right_frame
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        # Statistics table
        stats_frame = tk.Frame(self.right_frame, bg="white")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        columns = ("Metric", "Value")

        table = ttk.Treeview(
            stats_frame,
            columns=columns,
            show="headings",
            height=9
        )

        table.heading("Metric", text="Metric")
        table.heading("Value", text="Value")

        table.column("Metric", width=250)
        table.column("Value", width=250)

        stats_values = [
            ("Mean", f"{np.nanmean(valid_data):.4f}"),
            ("Median", f"{np.nanmedian(valid_data):.4f}"),
            ("Standard Deviation", f"{np.nanstd(valid_data):.4f}"),
            ("Minimum", f"{np.nanmin(valid_data):.4f}"),
            ("Maximum", f"{np.nanmax(valid_data):.4f}"),
            ("Variance", f"{np.nanvar(valid_data):.4f}"),
            ("25 Percentile", f"{np.nanpercentile(valid_data, 25):.4f}"),
            ("75 Percentile", f"{np.nanpercentile(valid_data, 75):.4f}"),
            ("Valid Pixels", f"{full_valid_count}")
        ]

        for row in stats_values:
            table.insert("", tk.END, values=row)

        table.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Class percentage table
        class_table = ttk.Treeview(
            stats_frame,
            columns=("Class", "Pixels", "Percentage"),
            show="headings",
            height=5
        )

        class_table.heading("Class", text="Class")
        class_table.heading("Pixels", text="Pixels")
        class_table.heading("Percentage", text="Percentage (%)")

        class_table.column("Class", width=180)
        class_table.column("Pixels", width=150)
        class_table.column("Percentage", width=150)

        total = sum(pie_values)

        class_rows = [
            ("Very Low", very_low),
            ("Low", low),
            ("Moderate", moderate),
            ("High", high)
        ]

        for cname, pixels in class_rows:
            pct = (pixels / total) * 100 if total > 0 else 0
            class_table.insert(
                "",
                tk.END,
                values=(cname, pixels, f"{pct:.2f}")
            )

        class_table.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.status.config(
            text=f"{title} computed successfully"
        )

    # ========================================================
    # SAVE CURRENT GEOTIFF
    # ========================================================

    def save_current_geotiff(self):

        if self.last_result is None:
            messagebox.showwarning(
                "No Result",
                "Please compute an index first."
            )
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            initialfile=f"{self.last_title}.tif",
            filetypes=[("GeoTIFF", "*.tif")]
        )

        if not output_path:
            return

        profile = self.reference_profile.copy()

        profile.update(
            dtype="float32",
            count=1,
            compress="lzw",
            nodata=np.nan
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(self.last_result, 1)

        messagebox.showinfo(
            "Saved",
            f"Saved successfully:\n{output_path}"
        )

    # ========================================================
    # CLEAR DASHBOARD
    # ========================================================

    def clear_dashboard(self):

        for widget in self.right_frame.winfo_children():
            widget.destroy()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app = RemoteSensingApp()
    app.mainloop()