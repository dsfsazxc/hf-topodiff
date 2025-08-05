import os
import argparse
import pickle
from functools import partial
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.font as tkfont
import tkinter.messagebox as msgbox
import tkinter.filedialog as filedialog
import numpy as np
import math
import shutil
from skimage import measure
import re

def center_crop_arr(pil_image, image_size):
    """Center crop and resize image to target size"""
    # Apply BOX downsampling first (powers of 2)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    # Calculate scale and apply BICUBIC resize
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    # Center crop
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

class CompositeImageButton(tk.Frame):
    def __init__(self, master, img_paths, resolution, bc_data=None, 
                 global_idx=None, data_path=None, idx_offset=800, cons_offset=200, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.img_paths = img_paths
        self.bc_data = bc_data
        self.global_idx = global_idx
        self.data_path = data_path
        self.idx_offset = idx_offset
        self.cons_offset = cons_offset

        self.resolution = resolution // 2
        self.padding = 8
        self.bottom_offset = 25

        # BC and contour box sizes - same as image size
        self.bc_box_width = self.resolution
        self.bc_box_height = self.resolution
        self.contour_box_width = self.resolution
        self.contour_box_height = self.resolution
        
        # Canvas size
        self.canvas_size = self.resolution * 2 + self.padding * 12 + self.bc_box_width + self.padding
        self.canvas_height = self.resolution * 2 + self.padding * 3 + self.bottom_offset + 50

        # Button states for each metric (0: inactive, 1: active)
        self.button_states = {
            'BC': [0, 0, 0, 0],  # Boundary Condition violation
            'FM': [0, 0, 0, 0],  # Floating material
        }

        # Store metric text and button background IDs
        self.text_ids = {
            'BC': [None, None, None, None],
            'FM': [None, None, None, None],
        }
        
        self.button_bg_ids = {
            'BC': [None, None, None, None],
            'FM': [None, None, None, None],
        }

        self.canvas = tk.Canvas(
            self,
            width=self.canvas_size,
            height=self.canvas_height,
            highlightthickness=0,
            bg="white"
        )
        self.canvas.grid(row=0, column=0)

        # Load constraint data
        self.obj_vf = None
        self.contour_data = None
        if self.global_idx is not None and self.data_path is not None and bc_data is not None:
            self.load_constraint_data()

        self.update_images()
        
    def load_constraint_data(self):
        """Load constraint data from numpy files"""
        try:
            file_path = os.path.join(self.data_path, f"cons_pf_array_{self.global_idx+self.cons_offset}.npy")
            if os.path.exists(file_path):
                data = np.load(file_path)
                # Obj VF: first channel's first value
                self.obj_vf = data[0, 0, 0]
                # Contour: third channel (Channel 2)
                self.contour_data = data[:, :, 2]
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error loading constraint data: {e}")
            self.obj_vf = None
            self.contour_data = None
    
    def update_images(self):
        """Update and display images with UI elements"""
        self.images = []
        for p in self.img_paths:
            if p is not None:
                img = Image.open(p).resize((self.resolution, self.resolution))
            else:
                img = Image.new("RGB", (self.resolution, self.resolution), "white")
            self.images.append(img)
        self.tk_images = [ImageTk.PhotoImage(img) for img in self.images]
        self.canvas.delete("all")

        # BC box position
        bc_box_x = self.padding
        bc_box_y = self.padding
        
        # Contour box position (below BC box)
        contour_box_x = bc_box_x
        contour_box_y = bc_box_y + self.bc_box_height + self.padding + 30
        
        # Image grid start position
        image_grid_x = bc_box_x + self.bc_box_width + self.padding * 2
        
        # 4 image coordinates
        self.coords = [
            (image_grid_x + 20, self.padding),
            (image_grid_x + self.resolution + self.padding * 8, self.padding),
            (image_grid_x + 20, self.resolution + 2 * self.padding + self.bottom_offset+10),
            (image_grid_x + self.resolution + self.padding * 8, self.resolution + 2 * self.padding + self.bottom_offset+10)
        ]

        coords = self.coords
        # Draw 4 images with gray borders
        for i, (x, y) in enumerate(coords):
            self.canvas.create_rectangle(
                x-1, y-1, 
                x+self.resolution+1, y+self.resolution+1,
                outline="#d0d0d0", width=1, tags="gray_border"
            )
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_images[i])

        # Create metric buttons (BC, FM only)
        for idx in range(len(coords)):
            cx = coords[idx][0] + self.resolution // 2
            cy = coords[idx][1] + self.resolution + 5
            
            # Split BC and FM buttons
            button_width = self.resolution // 2 - 2
            button_height = 30
            
            # BC button (left)
            bc_x = coords[idx][0]
            bc_y = cy
            self.create_clean_button(bc_x, bc_y, button_width, button_height, "BC", "BC", idx)
            
            # FM button (right)
            fm_x = coords[idx][0] + self.resolution // 2 + 2
            fm_y = cy
            self.create_clean_button(fm_x, fm_y, button_width, button_height, "FM", "FM", idx)

        # Draw BC box (black border, white background)
        self.canvas.create_rectangle(
            bc_box_x, bc_box_y, 
            bc_box_x + self.bc_box_width, bc_box_y + self.bc_box_height,
            outline="black", width=2, fill="white"
        )
        
        # BC box title
        self.canvas.create_text(
            bc_box_x + self.bc_box_width/2, bc_box_y + 12,
            text="BC & Load",
            font=("Arial", 60, "bold"),
            fill="black"
        )
        
        # Draw contour box (black border, white background)
        self.canvas.create_rectangle(
            contour_box_x, contour_box_y,
            contour_box_x + self.contour_box_width, contour_box_y + self.contour_box_height,
            outline="black", width=1, fill="white"
        )
        
        # Display boundary conditions and loads
        x_load_val = None
        y_load_val = None
        
        if self.bc_data is not None:
            grid_size = 64
            num_nodes = grid_size + 1

            # Draw boundary conditions
            bc_conf = self.bc_data.get('BC_conf', None)
            if bc_conf is not None:
                color_map = {1: "blue", 2: "red", 3: "green"}
                for node_list, direction in bc_conf:
                    color = color_map.get(direction, "gray")
                    for node in node_list:
                        iy = (node - 1) // num_nodes
                        ix = (node - 1) % num_nodes

                        # Transform coordinates: (ix, iy) -> (iy, grid_size - ix)
                        new_ix = iy
                        new_iy = ix

                        # Calculate position in BC box
                        px = bc_box_x + (new_ix / grid_size) * self.bc_box_width
                        py = bc_box_y + (new_iy / grid_size) * self.bc_box_height

                        size = 4
                        self.canvas.create_rectangle(
                            px - size, py - size, px + size, py + size,
                            fill=color, outline=color
                        )

            # Draw load arrows
            try:
                load_coord = self.bc_data.get('load_coord', None)
                x_loads = self.bc_data.get('x_loads', None)
                y_loads = self.bc_data.get('y_loads', None)
                if load_coord is not None and x_loads is not None and y_loads is not None:
                    norm_x = load_coord[0][0]
                    norm_y = load_coord[0][1]
                    dx = x_loads[0]
                    dy = y_loads[0]
                    
                    x_load_val = dx
                    y_load_val = dy

                    # Calculate load position in BC box
                    start_x = bc_box_x + norm_x * self.bc_box_width
                    start_y = bc_box_y + (1 - norm_y) * self.bc_box_height

                    vec_len = (dx ** 2 + dy ** 2) ** 0.5
                    if vec_len < 1e-6:
                        dx, dy = 0, -1

                    arrow_scale = 15
                    end_x = start_x + dx * arrow_scale
                    end_y = start_y + dy * arrow_scale

                    dot = self.canvas.create_oval(
                        start_x - 3, start_y - 3, start_x + 3, start_y + 3,
                        fill="purple", outline="purple", width=1
                    )
                    arrow = self.canvas.create_line(
                        start_x, start_y,
                        end_x, end_y,
                        arrow=tk.LAST,
                        width=3,
                        fill="purple",
                        capstyle=tk.ROUND,
                        joinstyle=tk.ROUND
                    )

                    self.canvas.tag_raise(dot)
                    self.canvas.tag_raise(arrow)

            except Exception as e:
                print("Load draw error:", e)
        
        # Display Load and Obj VF values
        text_y = bc_box_y + self.bc_box_height - 60
        if x_load_val is not None and y_load_val is not None:
            self.canvas.create_text(
                bc_box_x + self.bc_box_width/2, text_y,
                text=f"Fx: {x_load_val:.2f}",
                font=("Arial", 18),
                fill="black"
            )
            self.canvas.create_text(
                bc_box_x + self.bc_box_width/2, text_y + 18,
                text=f"Fy: {-y_load_val:.2f}",
                font=("Arial", 18),
                fill="black"
            )
        
        if self.obj_vf is not None:
            self.canvas.create_text(
                bc_box_x + self.bc_box_width/2, text_y + 36,
                text=f"VF: {self.obj_vf:.2f}",
                font=("Arial", 18, "bold"),
                fill="black"
            )
        
        # Draw contours using marching squares
        if self.contour_data is not None:
            try:
                # Normalize data
                data = self.contour_data
                vmin, vmax = np.percentile(data, 0), np.percentile(data, 99)
                if vmax > vmin:
                    normalized_data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
                else:
                    normalized_data = np.zeros_like(data)

                # Grid and cell size
                grid_size = normalized_data.shape[0]
                cell_width = self.contour_box_width / grid_size
                cell_height = self.contour_box_height / grid_size

                # Define levels
                levels = [i/30.0 for i in range(1, 30)]

                # Draw contours for each level
                for level in levels:
                    # Color calculation (red->blue gradient)
                    r = int(level * 255)
                    b = int((1-level) * 255)
                    g = 0
                    if 0.3 < level < 0.7:
                        g = int((0.5 - abs(level - 0.5)) * 100)
                    r, g, b = map(lambda v: max(0, min(255, v)), (r, g, b))
                    color = f'#{r:02x}{g:02x}{b:02x}'

                    # Extract marching squares contours
                    contours = measure.find_contours(normalized_data, level)
                    for contour in contours:
                        pts = []
                        for y_idx, x_idx in contour:
                            px = contour_box_x + x_idx * cell_width
                            py = contour_box_y + y_idx * cell_height
                            pts.extend((px, py))
                        if len(pts) >= 4:
                            self.canvas.create_line(*pts, fill=color, width=1, smooth=True)

                # Add title
                self.canvas.create_text(
                    contour_box_x + self.contour_box_width/2,
                    contour_box_y + self.contour_box_height + 10,
                    text="Strain energy",
                    font=("Arial", 32, "bold"),
                    fill="blue",
                    anchor="center"
                )

                # Draw colorbar
                colorbar_x = contour_box_x
                colorbar_y = contour_box_y + self.contour_box_height + 20
                colorbar_width = self.contour_box_width
                colorbar_height = 15

                for i in range(colorbar_width):
                    value = i / colorbar_width
                    # Same color mapping as above
                    r = int(value * 255)
                    b = int((1-value) * 255)
                    g = 0
                    if 0.3 < value < 0.7:
                        g = int((0.5 - abs(value - 0.5)) * 100)
                    r, g, b = map(lambda v: max(0, min(255, v)), (r, g, b))
                    color = f'#{r:02x}{g:02x}{b:02x}'

                    self.canvas.create_line(
                        colorbar_x + i, colorbar_y,
                        colorbar_x + i, colorbar_y + colorbar_height,
                        fill=color, width=1
                    )

                # Low/High labels
                self.canvas.create_text(
                    colorbar_x, colorbar_y + colorbar_height + 5,
                    text="Low", font=("Arial", 8), fill="black", anchor="w"
                )
                self.canvas.create_text(
                    colorbar_x + colorbar_width, colorbar_y + colorbar_height + 5,
                    text="High", font=("Arial", 8), fill="black", anchor="e"
                )

            except Exception as e:
                print(f"Contour drawing error: {e}")

        # Update button states
        self.update_button_states()

    def create_clean_button(self, x, y, width, height, text, metric_type, idx):
        """Create stylized button"""
        # Shadow for gradient effect
        shadow_id = self.canvas.create_rectangle(
            x + 2, y + 2, x + width + 2, y + height + 2,
            fill="#d0d0d0", outline="", width=0,
            tags=f"{metric_type.lower()}_button_shadow_{idx}"
        )
        
        # Button background (rounded corner effect)
        button_id = self.canvas.create_rectangle(
            x, y, x + width, y + height,
            fill="#f8f8f8", outline="#c0c0c0", width=2,
            tags=f"{metric_type.lower()}_button_bg_{idx}"
        )
        
        # Inner highlight (3D effect)
        highlight_id = self.canvas.create_rectangle(
            x + 1, y + 1, x + width - 1, y + height // 2,
            fill="#ffffff", outline="", width=0,
            tags=f"{metric_type.lower()}_button_highlight_{idx}"
        )
        
        # Text
        text_id = self.canvas.create_text(
            x + width // 2, y + height // 2,
            text=text,
            font=("Arial", 32, "bold"),  
            fill="#333333",
            anchor="center",
            tags=f"{metric_type.lower()}_text_{idx}"
        )
        
        # Store IDs
        self.button_bg_ids[metric_type][idx] = button_id
        self.text_ids[metric_type][idx] = text_id
        
        # Bind click events
        for element_id in [button_id, highlight_id, text_id]:
            self.canvas.tag_bind(
                element_id, 
                "<Button-1>", 
                lambda event, m=metric_type, i=idx: self.toggle_button_state(m, i)
            )
            
            # Mouse hover effects
            self.canvas.tag_bind(
                element_id,
                "<Enter>",
                lambda event, bid=button_id: self.canvas.itemconfig(bid, fill="#f0f0f0")
            )
            self.canvas.tag_bind(
                element_id,
                "<Leave>",
                lambda event, bid=button_id, m=metric_type, i=idx: self.update_single_button_state(m, i)
            )

    def toggle_button_state(self, metric_type, idx):
        """Toggle button state (0 -> 1, 1 -> 0)"""
        self.button_states[metric_type][idx] = 1 - self.button_states[metric_type][idx]
        self.update_button_states()

    def update_single_button_state(self, metric_type, idx):
        """Update single button state (for hover effects)"""
        state = self.button_states[metric_type][idx]
        button_id = self.button_bg_ids[metric_type][idx]
        
        if button_id is not None:
            if state == 1:  # Selected state
                color_map = {'BC': '#d0ffd0', 'FM': '#ffd0d0'}
                self.canvas.itemconfig(button_id, fill=color_map[metric_type])
            else:  # Unselected state
                self.canvas.itemconfig(button_id, fill="#f8f8f8")

    def update_image_borders(self):
        """Update image borders based on button states"""
        coords = self.coords
        for i, (x, y) in enumerate(coords):
            self.canvas.delete(f"img_border_{i}")
            bc = self.button_states['BC'][i]
            fm = self.button_states['FM'][i]
            if bc and fm:
                border_color = "#8000ff"
            elif bc:
                border_color = "#0066ff"
            elif fm:
                border_color = "#ff3333"
            else:
                border_color = "#d0d0d0"
            self.canvas.create_rectangle(
                x-1, y-1, x+self.resolution+1, y+self.resolution+1,
                outline=border_color, width=5, tags=f"img_border_{i}"
            )

    def update_button_states(self):
        """Update button appearance based on states"""
        color_map = {
            'BC': 'blue',  
            'FM': 'red',  
        }
        bg_color_map = {
            'BC': '#d0ffd0',
            'FM': '#ffd0d0',
        }

        self.update_image_borders()

        for metric_type in self.button_states:
            for idx, state in enumerate(self.button_states[metric_type]):
                button_id = self.button_bg_ids[metric_type][idx]
                text_id   = self.text_ids[metric_type][idx]
                if button_id is None or text_id is None:
                    continue

                if state == 1:
                    # Highlight background
                    self.canvas.itemconfig(
                        button_id,
                        fill=bg_color_map[metric_type],
                        outline=color_map[metric_type],
                        width=3
                    )
                    # Change text color & style
                    self.canvas.itemconfig(
                        text_id,
                        fill=color_map[metric_type],
                        font=("Arial", 20, "bold", "underline")
                    )
                else:
                    # Reset to default
                    self.canvas.itemconfig(
                        button_id,
                        fill="#f8f8f8",
                        outline="#c0c0c0",
                        width=2
                    )
                    self.canvas.itemconfig(
                        text_id,
                        fill="#333333",
                        font=("Arial", 20, "bold")
                    )

    def get_button_states(self):
        """Return current button states"""
        return self.button_states

    def reset(self):
        """Reset all button states"""
        for metric_type in self.button_states:
            self.button_states[metric_type] = [0, 0, 0, 0]
        self.update_button_states()

def _list_image_files_sorted(folder):
    """List and sort image files by number in filename"""
    def extract_number(filename):
        match = re.search(r'\d+', os.path.splitext(filename)[0])
        return int(match.group()) if match else -1

    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".png", ".jpg", ".jpeg"))
    ], key=lambda f: extract_number(os.path.basename(f)))

def main():
    args = create_argparser().parse_args()

    # Process data directories
    data_dirs = [args.data_dir1, args.data_dir2, args.data_dir3, args.data_dir4]

    # data_dir1 is required, create None lists for missing directories
    main_list = _list_image_files_sorted(args.data_dir1)
    list_files = []
    for d in data_dirs:
        if d is not None:
            list_files.append(_list_image_files_sorted(d))
        else:
            list_files.append([None] * len(main_list))

    # Group images based on data_dir1 count (4 images per group)
    all_image_groups = list(zip(*list_files))
    total_groups = len(all_image_groups)
    page_size = args.grid_row * args.grid_col
    current_page_index = 0

    # Process BC_dir4: boundary condition data (.npy file)
    if args.BC_dir4 and os.path.isfile(args.BC_dir4):
        bc_array = np.load(args.BC_dir4, allow_pickle=True)
    else:
        bc_array = None

    # Constraint data path
    constraint_data_path = args.constraint_data_path

    # Initialize feedback dictionary for feedback model
    feedbacks = {
        'BC': {},  # Boundary Condition violation
        'FM': {},  # Floating material
    }

    # Load previous feedbacks if available
    if args.feedback_path and os.path.isfile(args.feedback_path):
        try:
            with open(args.feedback_path, "rb") as f:
                old_feedbacks = pickle.load(f)
                # Convert old format to new format if needed
                if isinstance(old_feedbacks, dict):
                    if not any(isinstance(val, dict) for val in old_feedbacks.values()):
                        # Convert old format to new structure
                        for img_path, value in old_feedbacks.items():
                            if value == 0:  # Red (floating material)
                                feedbacks['FM'][img_path] = 1
                            elif value == 3:  # Green (boundary condition violation)
                                feedbacks['BC'][img_path] = 1
                    else:
                        # Already new format, use directly (BC and FM only)
                        if 'BC' in old_feedbacks:
                            feedbacks['BC'] = old_feedbacks['BC']
                        elif 'LV' in old_feedbacks:  # Convert old LV to BC
                            feedbacks['BC'] = old_feedbacks['LV']
                        if 'FM' in old_feedbacks:
                            feedbacks['FM'] = old_feedbacks['FM']
        except Exception as e:
            print(f"Error loading feedback file: {e}")

    root = tk.Tk()
    root.title("Feedback data collector")

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack()

    # Horizontal frame
    question_row = tk.Frame(frame)
    question_row.pack(fill="x", expand=True)

    # Instructions (left)
    question = tk.Label(
        question_row,
        text="Click on metrics to select:\nBC = GREEN (Boundary Condition violation)\nFM = RED (Floating material)",
        font=tkfont.Font(family="Arial", size=32, weight="bold"),
        justify="left",
        pady=10
    )
    question.pack(side="left", padx=(0, 20))

    # Legend (right) - horizontal layout
    legend_frame = tk.Frame(question_row)
    legend_frame.pack(side="right", padx=(10, 0))

    def make_color_box(parent, color, text):
        item = tk.Frame(parent)
        color_patch = tk.Canvas(item, width=15, height=15, highlightthickness=0)
        color_patch.create_rectangle(0, 0, 15, 15, fill=color, outline="gray")
        color_patch.pack(side="left", padx=(0, 4))
        label = tk.Label(item, text=text, font=("Arial", 32))
        label.pack(side="left")
        return item
    
    def make_color_box_frame(parent, color, text):
        item = tk.Frame(parent)
        color_patch = tk.Canvas(item, width=15, height=15, highlightthickness=0)
        color_patch.create_rectangle(1, 1, 14, 14, fill="", outline=color, width=2)
        color_patch.pack(side="left", padx=(0, 4))
        label = tk.Label(item, text=text, font=("Arial", 32))
        label.pack(side="left")
        return item
    
    # Selection legend - horizontal layout
    selection_legend_frame = tk.Frame(legend_frame)
    selection_legend_frame.pack(side="top", anchor="e", pady=(0, 5))
    
    # Boundary condition legend - horizontal layout
    bc_legend_frame = tk.Frame(legend_frame)
    bc_legend_frame.pack(side="top", anchor="e")
    
    bc_legend_label = tk.Label(bc_legend_frame, text="Boundary conditions:", font=("Arial", 32, "bold"))
    bc_legend_label.pack(side="left", padx=(0, 5))
    
    make_color_box(bc_legend_frame, "purple", "Load").pack(side="left", padx=(0, 5))
    make_color_box(bc_legend_frame, "blue", "X-fixed").pack(side="left", padx=(0, 5))
    make_color_box(bc_legend_frame, "red", "Y-fixed").pack(side="left", padx=(0, 5))
    make_color_box(bc_legend_frame, "green", "X,Y-fixed").pack(side="left")
    make_color_box_frame(selection_legend_frame, "#0066ff", "Boundary Condition violation (BC)").pack(side="left", padx=(0, 10))
    make_color_box_frame(selection_legend_frame, "#ff3333", "Floating material (FM)").pack(side="left")
    make_color_box_frame(selection_legend_frame, "#8000ff", "Both BC & FM").pack(side="left")

    img_grid = tk.Frame(frame)
    img_grid.pack(pady=10)

    image_buttons = []

    def populate_page():
        nonlocal current_page_index, image_buttons
        for widget in img_grid.winfo_children():
            widget.destroy()
        image_buttons.clear()
        start = current_page_index * page_size
        end = min(start + page_size, total_groups)
        
        for local_idx, group in enumerate(all_image_groups[start:end]):
            global_idx = start + local_idx + args.idx_offset
            
            # All images in group use same boundary condition (bc_value)
            bc_value = bc_array[global_idx] if bc_array is not None and len(bc_array) > global_idx else None
            
            btn = CompositeImageButton(
                img_grid, 
                img_paths=list(group), 
                resolution=args.resolution, 
                bc_data=bc_value,
                global_idx=global_idx,
                data_path=constraint_data_path,
                idx_offset=args.idx_offset,
                cons_offset=args.cons_offset
            )
            
            # Apply existing feedbacks if available
            for metric_type in feedbacks:
                for idx, img_path in enumerate(group):
                    if img_path is not None and img_path in feedbacks[metric_type]:
                        btn.button_states[metric_type][idx] = feedbacks[metric_type].get(img_path, 0)
            
            btn.update_button_states()
            btn.grid(row=local_idx // args.grid_col, column=local_idx % args.grid_col, padx=5, pady=5)
            image_buttons.append((btn, group, global_idx))
        update_page_info()

    def update_page_info():
        page_info.config(text=f"Page {current_page_index + 1} / {math.ceil(total_groups / page_size)}")

    def store_response_and_next():
        nonlocal current_page_index
        for btn, group, global_idx in image_buttons:
            states = btn.get_button_states()
            for metric_type in states:
                for i, img_path in enumerate(group):
                    if img_path is not None:
                        # Save only if state is 1 (selected)
                        if states[metric_type][i] == 1:
                            feedbacks[metric_type][img_path] = 1
                        # Remove if state is 0 (unselected) and was previously selected
                        elif img_path in feedbacks[metric_type]:
                            del feedbacks[metric_type][img_path]
                            
        if (current_page_index + 1) * page_size < total_groups:
            current_page_index += 1
            populate_page()
        else:
            msgbox.showinfo("End", "All images have been evaluated.")

    def go_previous():
        nonlocal current_page_index
        if current_page_index > 0:
            current_page_index -= 1
            populate_page()
        else:
            msgbox.showinfo("Info", "This is the first page.")

    def reset_selections():
        for btn, group, _ in image_buttons:
            btn.reset()
            
    def save_feedbacks(quit_program=False):
        """Save feedback dataset"""
        output_dir = filedialog.askdirectory(
            title="Select directory to save feedback dataset",
            initialdir=os.path.dirname(args.feedback_output_dir) if args.feedback_output_dir else None
        )
        
        if not output_dir:
            if quit_program:
                root.quit()
                root.destroy()
            return
        
        # Create feedback data directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect feedback states as arrays (BC and FM only)
        bc_feedback = []
        fm_feedback = []
        
        # Image path and index mapping
        image_map = {}
        next_idx = 0
        
        # Process all image groups
        for gi, grp in enumerate(all_image_groups):
            if not any(grp):
                continue
            
            gid = gi + args.idx_offset
            
            # Check boundary condition data
            if bc_array is None or gid >= len(bc_array):
                continue
                
            bc_val = bc_array[gid]
            
            # Constraint data paths for each group
            pf_path = os.path.join(constraint_data_path, f"cons_pf_array_{gid+args.cons_offset}.npy")
            load_path = os.path.join(constraint_data_path, f"cons_load_array_{gid+args.cons_offset}.npy")
            bc_path = os.path.join(constraint_data_path, f"cons_bc_array_{gid+args.cons_offset}.npy")
            
            if not (os.path.exists(pf_path) and os.path.exists(load_path) and os.path.exists(bc_path)):
                continue
                
            # Load constraint data
            try:
                constraints_pf = np.load(pf_path)
                loads = np.load(load_path)
                bcs = np.load(bc_path)
            except Exception as e:
                print(f"Error loading constraint data for group {gid}: {e}")
                continue
                
            # Process each image
            for idx, img_path in enumerate(grp):
                if not img_path:
                    continue
                    
                # Check feedback info for current image (BC and FM only)
                is_bc = img_path in feedbacks['BC']
                is_fm = img_path in feedbacks['FM']
                
                # Add to feedback arrays (BC and FM only)
                bc_feedback.append(1 if is_bc else 0)
                fm_feedback.append(1 if is_fm else 0)
                
                # Image save path
                dest_img_path = os.path.join(output_dir, f"gt_topo_{next_idx}.png")
                
                # Copy image
                try:
                    shutil.copy2(img_path, dest_img_path)
                except Exception as e:
                    print(f"Error copying image {img_path}: {e}")
                    continue
                
                # Save constraint data
                np.save(os.path.join(output_dir, f"cons_pf_array_{next_idx}.npy"), constraints_pf)
                np.save(os.path.join(output_dir, f"cons_load_array_{next_idx}.npy"), loads)
                np.save(os.path.join(output_dir, f"cons_bc_array_{next_idx}.npy"), bcs)
                
                # Update image mapping
                image_map[img_path] = next_idx
                next_idx += 1
        
        # Save feedback info as .npy files (BC and FM only)
        np.save(os.path.join(output_dir, "feedback_bc.npy"), np.array(bc_feedback, dtype=np.int8))
        np.save(os.path.join(output_dir, "feedback_fm.npy"), np.array(fm_feedback, dtype=np.int8))
        
        # Create combined feedback file (single label for training)
        # Combine BC and FM into single value per image
        # 0: normal, 1: BC issue, 2: FM issue, 3: both BC and FM issues
        combined_feedback = []
        for i in range(len(bc_feedback)):
            value = 0
            if bc_feedback[i] == 1:
                value += 1
            if fm_feedback[i] == 1:
                value += 2
            combined_feedback.append(value)
        
        np.save(os.path.join(output_dir, "feedback_combined.npy"), np.array(combined_feedback, dtype=np.int8))
        
        # Save summary info
        with open(os.path.join(output_dir, "feedback_summary.txt"), "w") as f:
            f.write(f"Total images: {next_idx}\n")
            f.write(f"BC (Boundary Condition violation) positive: {sum(bc_feedback)}\n")
            f.write(f"FM (Floating material) positive: {sum(fm_feedback)}\n")
            f.write(f"Both BC and FM positive: {sum([1 for bc, fm in zip(bc_feedback, fm_feedback) if bc == 1 and fm == 1])}\n")
            f.write(f"No issues: {sum([1 for bc, fm in zip(bc_feedback, fm_feedback) if bc == 0 and fm == 0])}\n")
        
        # Save image mapping info (for debugging)
        with open(os.path.join(output_dir, "image_mapping.pkl"), "wb") as f:
            pickle.dump(image_map, f)
        
        # Completion message
        msgbox.showinfo(
            "Save complete",
            f"Dataset saved successfully!\n"
            f"Location: {output_dir}\n"
            f"Total images: {next_idx}\n"
            f"BC (Boundary Condition violation) positive: {sum(bc_feedback)}\n"
            f"FM (Floating material) positive: {sum(fm_feedback)}\n"
            f"Both BC & FM positive: {sum([1 for bc, fm in zip(bc_feedback, fm_feedback) if bc == 1 and fm == 1])}\n"
            f"No issues: {sum([1 for bc, fm in zip(bc_feedback, fm_feedback) if bc == 0 and fm == 0])}"
        )
        
        if quit_program:
            root.quit()
            root.destroy()

    nav_frame = tk.Frame(frame)
    nav_frame.pack(pady=10)

    prev_btn = tk.Button(nav_frame, text="Previous", width=10, command=go_previous, font=("Arial", 12))
    prev_btn.grid(row=0, column=0, padx=5)

    reset_btn = tk.Button(nav_frame, text="Reset", width=10, command=reset_selections, font=("Arial", 12))
    reset_btn.grid(row=0, column=1, padx=5)

    submit_btn = tk.Button(nav_frame, text="Submit & Next", width=12, command=store_response_and_next, font=("Arial", 12))
    submit_btn.grid(row=0, column=2, padx=5)

    save_btn = tk.Button(nav_frame, text="Save", width=10, command=save_feedbacks, font=("Arial", 12))
    save_btn.grid(row=0, column=3, padx=5)

    page_info = tk.Label(frame, text="", font=("Arial", 12))
    page_info.pack()

    def on_close():
        # Show save confirmation dialog
        if msgbox.askyesno("Exit", "Do you want to save before exit?"):
            save_feedbacks(quit_program=True)  # Save and exit
        else:
            root.quit()
            root.destroy()  # Exit without saving
    
    # Bind ESC key and window close button
    root.bind("<Escape>", lambda event: on_close())
    root.protocol("WM_DELETE_WINDOW", on_close)

    populate_page()
    root.mainloop()

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir1", type=str, required=True, help="Primary image directory")
    parser.add_argument("--data_dir2", type=str, default=None, help="Secondary image directory")
    parser.add_argument("--data_dir3", type=str, default=None, help="Third image directory")
    parser.add_argument("--data_dir4", type=str, default=None, help="Fourth image directory")
    parser.add_argument("--BC_dir4", type=str, default=None, help="Boundary condition data file")
    parser.add_argument("--constraint_data_path", type=str, default=None,
                       help="Path to directory containing cons_pf_array_XXX.npy files")
    parser.add_argument("--feedback_path", type=str, default=None,
                       help="Path to existing feedback file to load")
    parser.add_argument("--feedback_output_dir", type=str, default=None,
                       help="Directory to output the feedback dataset")
    parser.add_argument("--censoring_feature", type=str, required=True, help="Censoring feature name")
    parser.add_argument("--resolution", type=int, default=150, help="Image display resolution")
    parser.add_argument("--grid_row", type=int, default=4, help="Grid rows per page")
    parser.add_argument("--grid_col", type=int, default=4, help="Grid columns per page")
    parser.add_argument("--idx_offset", type=int, default=800, help="Global index offset")
    parser.add_argument("--cons_offset", type=int, default=200, help="Constraint file offset")
    return parser

if __name__ == "__main__":
    main()