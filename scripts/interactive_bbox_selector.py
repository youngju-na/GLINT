#!/usr/bin/env python3
"""
Interactive 3D Bounding Box Selector using Open3D

Usage:
    python scripts/interactive_bbox_selector.py --mesh path/to/mesh.ply

Controls in viewer window:
    - Left click + drag: Rotate
    - Scroll: Zoom
    - Right click + drag: Pan
    - Press 'H' for help
"""

import argparse
import numpy as np
import open3d as o3d
import yaml
from pathlib import Path


def create_bbox_lineset(bbox_min, bbox_max, color=[1, 0, 0]):
    """Create a LineSet representing the bounding box edges"""
    points = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
    ])

    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical
    ])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

    return line_set


def create_sphere(center, radius, color=[1, 0, 0]):
    """Create a sphere mesh at the given center"""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


class BBoxSelector:
    def __init__(self, mesh, mesh_path):
        self.mesh = mesh
        self.mesh_path = Path(mesh_path)

        # Get mesh bounding box
        mesh_bbox = mesh.get_axis_aligned_bounding_box()
        self.mesh_min = np.array(mesh_bbox.min_bound)
        self.mesh_max = np.array(mesh_bbox.max_bound)
        self.mesh_center = (self.mesh_min + self.mesh_max) / 2
        self.mesh_size = self.mesh_max - self.mesh_min

        # Current bbox (start with mesh bounds)
        self.bbox_min = self.mesh_min.copy()
        self.bbox_max = self.mesh_max.copy()

        # Adjustment parameters
        self.current_axis = 2  # 0=X, 1=Y, 2=Z
        self.step = np.max(self.mesh_size) * 0.02
        self.adjusting_min = True  # True=min, False=max

        # Visualization
        self.vis = None
        self.bbox_lineset = None
        self.coord_frame = None

    def print_help(self):
        print("\n" + "="*60)
        print("BBox Selector - Keyboard Controls")
        print("="*60)
        print("Axis Selection:")
        print("  X/Y/Z    : Select axis to adjust")
        print("\nBound Selection:")
        print("  M        : Toggle between MIN and MAX bound")
        print("\nAdjustment:")
        print("  UP/DOWN  : Increase/decrease current bound (+/- step)")
        print("  +/-      : Increase/decrease step size")
        print("\nActions:")
        print("  R        : Reset bbox to mesh bounds")
        print("  C        : Preview cropped mesh")
        print("  S        : Save bbox config")
        print("  W        : Save cropped mesh")
        print("  H        : Show this help")
        print("  Q/ESC    : Quit")
        print("="*60)
        self.print_status()

    def print_status(self):
        axis_names = ['X', 'Y', 'Z']
        bound_name = 'MIN' if self.adjusting_min else 'MAX'
        print(f"\nAxis: {axis_names[self.current_axis]} | Bound: {bound_name} | Step: {self.step:.4f}")
        print(f"BBox Min: [{self.bbox_min[0]:.4f}, {self.bbox_min[1]:.4f}, {self.bbox_min[2]:.4f}]")
        print(f"BBox Max: [{self.bbox_max[0]:.4f}, {self.bbox_max[1]:.4f}, {self.bbox_max[2]:.4f}]")
        size = self.bbox_max - self.bbox_min
        print(f"Size:     [{size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}]")

    def update_bbox_visualization(self):
        """Update the bounding box visualization"""
        # Remove old bbox
        if self.bbox_lineset is not None:
            self.vis.remove_geometry(self.bbox_lineset, reset_bounding_box=False)

        # Create new bbox
        self.bbox_lineset = create_bbox_lineset(self.bbox_min, self.bbox_max, color=[1, 0, 0])
        self.vis.add_geometry(self.bbox_lineset, reset_bounding_box=False)
        self.vis.update_renderer()

    def _make_key_callback(self, key):
        """Create a callback function for a specific key"""
        def callback(vis):
            if key == ord('X') or key == ord('x'):
                self.current_axis = 0
                print(f"Selected axis: X")
                self.print_status()
            elif key == ord('Y') or key == ord('y'):
                self.current_axis = 1
                print(f"Selected axis: Y")
                self.print_status()
            elif key == ord('Z') or key == ord('z'):
                self.current_axis = 2
                print(f"Selected axis: Z")
                self.print_status()
            elif key == ord('M') or key == ord('m'):
                self.adjusting_min = not self.adjusting_min
                bound_name = 'MIN' if self.adjusting_min else 'MAX'
                print(f"Adjusting: {bound_name}")
                self.print_status()
            elif key == 265:  # UP arrow
                if self.adjusting_min:
                    self.bbox_min[self.current_axis] += self.step
                else:
                    self.bbox_max[self.current_axis] += self.step
                self.update_bbox_visualization()
                self.print_status()
            elif key == 264:  # DOWN arrow
                if self.adjusting_min:
                    self.bbox_min[self.current_axis] -= self.step
                else:
                    self.bbox_max[self.current_axis] -= self.step
                self.update_bbox_visualization()
                self.print_status()
            elif key == ord('+') or key == ord('='):
                self.step *= 2
                print(f"Step size: {self.step:.4f}")
            elif key == ord('-'):
                self.step /= 2
                print(f"Step size: {self.step:.4f}")
            elif key == ord('R') or key == ord('r'):
                self.bbox_min = self.mesh_min.copy()
                self.bbox_max = self.mesh_max.copy()
                self.update_bbox_visualization()
                print("Reset to mesh bounds")
                self.print_status()
            elif key == ord('C') or key == ord('c'):
                self.preview_crop()
            elif key == ord('S') or key == ord('s'):
                self.save_config()
            elif key == ord('W') or key == ord('w'):
                self.save_cropped_mesh()
            elif key == ord('H') or key == ord('h'):
                self.print_help()
            return False
        return callback

    def preview_crop(self):
        """Show cropped mesh in a new window"""
        print("\nPreviewing cropped mesh...")
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=self.bbox_min,
            max_bound=self.bbox_max
        )
        cropped = self.mesh.crop(bbox)
        print(f"  Cropped: {len(cropped.vertices)} vertices, {len(cropped.triangles)} triangles")

        # Show with coord frame and bbox
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=np.max(self.mesh_size) * 0.2,
            origin=self.mesh_center
        )
        bbox_vis = create_bbox_lineset(self.bbox_min, self.bbox_max, color=[0, 1, 0])

        o3d.visualization.draw_geometries(
            [cropped, coord_frame, bbox_vis],
            window_name="Cropped Preview (close to continue)"
        )

    def save_config(self):
        """Save bbox config to yaml"""
        output_path = self.mesh_path.parent / "trans_bbox_config.yaml"

        config = {
            'runner_cfg': {
                'visualizer_cfg': {
                    'trans_bbox_min': [round(float(x), 4) for x in self.bbox_min],
                    'trans_bbox_max': [round(float(x), 4) for x in self.bbox_max],
                }
            }
        }

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=None)

        print(f"\n{'='*50}")
        print(f"Config saved to: {output_path}")
        print(f"\nAdd to your config yaml:")
        print(f"{'='*50}")
        print(f"runner_cfg:")
        print(f"    visualizer_cfg:")
        print(f"        trans_bbox_min: {[round(float(x), 4) for x in self.bbox_min]}")
        print(f"        trans_bbox_max: {[round(float(x), 4) for x in self.bbox_max]}")
        print(f"{'='*50}")

    def save_cropped_mesh(self):
        """Save cropped mesh to file"""
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=self.bbox_min,
            max_bound=self.bbox_max
        )
        cropped = self.mesh.crop(bbox)
        cropped.compute_vertex_normals()

        output_path = self.mesh_path.parent / f"{self.mesh_path.stem}_cropped.ply"
        o3d.io.write_triangle_mesh(str(output_path), cropped)

        print(f"\nCropped mesh saved to: {output_path}")
        print(f"  Original: {len(self.mesh.vertices)} vertices")
        print(f"  Cropped: {len(cropped.vertices)} vertices")

    def run(self):
        """Run the interactive selector"""
        print("\n" + "="*60)
        print("Interactive BBox Selector")
        print("="*60)
        print(f"Mesh: {self.mesh_path}")
        print(f"  Vertices: {len(self.mesh.vertices)}")
        print(f"  Triangles: {len(self.mesh.triangles)}")
        print(f"  Bounds: {self.mesh_min} to {self.mesh_max}")
        print(f"  Size: {self.mesh_size}")
        print("\nPress 'H' in the viewer window for help")
        print("="*60)

        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="BBox Selector - Press H for help", width=1280, height=720)

        # Register key callbacks
        keys_to_register = [
            ord('X'), ord('x'), ord('Y'), ord('y'), ord('Z'), ord('z'),
            ord('M'), ord('m'),
            ord('+'), ord('='), ord('-'),
            ord('R'), ord('r'),
            ord('C'), ord('c'),
            ord('S'), ord('s'),
            ord('W'), ord('w'),
            ord('H'), ord('h'),
            264, 265,  # DOWN, UP arrows
        ]
        for key in keys_to_register:
            self.vis.register_key_callback(key, self._make_key_callback(key))

        # Add mesh
        self.vis.add_geometry(self.mesh)

        # Add coordinate frame
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=np.max(self.mesh_size) * 0.2,
            origin=self.mesh_center
        )
        self.vis.add_geometry(self.coord_frame)

        # Add initial bbox
        self.bbox_lineset = create_bbox_lineset(self.bbox_min, self.bbox_max, color=[1, 0, 0])
        self.vis.add_geometry(self.bbox_lineset)

        # Set view
        ctr = self.vis.get_view_control()
        ctr.set_lookat(self.mesh_center)
        ctr.set_zoom(0.5)

        self.print_help()

        # Run
        self.vis.run()
        self.vis.destroy_window()

        return self.bbox_min.tolist(), self.bbox_max.tolist()


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D Bounding Box Selector")
    parser.add_argument("--mesh", "-m", type=str, required=True,
                        help="Path to mesh file (.ply)")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"Error: Mesh file not found: {mesh_path}")
        return

    print(f"Loading mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    print(f"Loaded: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    selector = BBoxSelector(mesh, mesh_path)
    bbox_min, bbox_max = selector.run()

    print(f"\nFinal BBox:")
    print(f"  Min: {bbox_min}")
    print(f"  Max: {bbox_max}")


if __name__ == "__main__":
    main()
