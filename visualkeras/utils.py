from typing import Any, Dict, Mapping, Union, Sequence, List, Tuple, Optional
from PIL import ImageColor, ImageDraw, Image, ImageFont
import aggdraw
import numpy as np
import math

def resolve_style(
    target: Any, 
    name: str, 
    styles: Mapping[Union[str, type], Dict[str, Any]], 
    defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generic style resolver.
    """
    final_style = defaults.copy()
    
    for cls in type(target).__mro__:
        if cls in styles:
            final_style.update(styles[cls])
    
    if name in styles:
        final_style.update(styles[name])
        
    return final_style


class RectShape:
    x1: int
    x2: int
    y1: int
    y2: int
    _fill: Any
    _outline: Any
    style: dict = None 

    def __init__(self):
        self.style = {}

    @property
    def fill(self):
        return self._fill

    @property
    def outline(self):
        return self._outline

    @fill.setter
    def fill(self, v):
        self._fill = get_rgba_tuple(v)

    @outline.setter
    def outline(self, v):
        self._outline = get_rgba_tuple(v)

    def _get_pen_brush(self):
        pen = aggdraw.Pen(self._outline)
        brush = aggdraw.Brush(self._fill)
        return pen, brush


class Box(RectShape):
    de: int
    shade: int
    rotation: Optional[float] = None  # Rotation around Y axis in degrees
    
    # Cache for projected faces to ensure logos/images align perfectly
    _projected_faces: Dict[int, List[Tuple[float, float]]] = None

    def get_face_quad(self, face_index: int) -> List[Tuple[float, float]]:
        """
        Returns the 4 projected screen coordinates [(x,y), ...] for the specified face.
        Face Indices:
        0: Front, 1: Back, 2: Right, 3: Left, 4: Top, 5: Bottom
        """
        if self._projected_faces and face_index in self._projected_faces:
            return self._projected_faces[face_index]
        return []

    def draw(self, draw: ImageDraw, draw_reversed: bool = False):
        pen, brush = self._get_pen_brush()
        
        # Dimensions
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        # Use 'de' as the Z-depth. 
        # In the layout, de was a shift offset. We treat it as physical depth here.
        d = getattr(self, "de", 0)
        
        if d == 0:
            # Fallback for flat nodes (2D)
            draw.rectangle([self.x1, self.y1, self.x2, self.y2], pen, brush)
            self._projected_faces = {0: [(self.x1, self.y1), (self.x2, self.y1), (self.x2, self.y2), (self.x1, self.y2)]}
            return

        if self.rotation is None:
            # Legacy Drawing Logic (Isometric-ish offset)
            if hasattr(self, 'de') and self.de > 0:
                brush_s1 = aggdraw.Brush(fade_color(self.fill, self.shade))
                brush_s2 = aggdraw.Brush(fade_color(self.fill, 2 * self.shade))

                if draw_reversed:
                    draw.line([self.x2 - self.de, self.y1 - self.de, self.x2 - self.de, self.y2 - self.de], pen)
                    draw.line([self.x2 - self.de, self.y2 - self.de, self.x2, self.y2], pen)
                    draw.line([self.x1 - self.de, self.y2 - self.de, self.x2 - self.de, self.y2 - self.de], pen)

                    draw.polygon([self.x1, self.y1,
                                  self.x1 - self.de, self.y1 - self.de,
                                  self.x2 - self.de, self.y1 - self.de,
                                  self.x2, self.y1
                                  ], pen, brush_s1)

                    draw.polygon([self.x1 - self.de, self.y1 - self.de,
                                  self.x1, self.y1,
                                  self.x1, self.y2,
                                  self.x1 - self.de, self.y2 - self.de
                                  ], pen, brush_s2)
                    
                    # Populate projected faces for legacy mode
                    self._projected_faces = {
                        0: [(self.x1, self.y1), (self.x2, self.y1), (self.x2, self.y2), (self.x1, self.y2)], # Front
                        4: [(self.x1 - self.de, self.y1 - self.de), (self.x2 - self.de, self.y1 - self.de), (self.x2, self.y1), (self.x1, self.y1)], # Top
                        2: [(self.x1 - self.de, self.y1 - self.de), (self.x1, self.y1), (self.x1, self.y2), (self.x1 - self.de, self.y2 - self.de)] # Side (Left)
                    }
                else:
                    draw.line([self.x1 + self.de, self.y1 - self.de, self.x1 + self.de, self.y2 - self.de], pen)
                    draw.line([self.x1 + self.de, self.y2 - self.de, self.x1, self.y2], pen)
                    draw.line([self.x1 + self.de, self.y2 - self.de, self.x2 + self.de, self.y2 - self.de], pen)

                    draw.polygon([self.x1, self.y1,
                                  self.x1 + self.de, self.y1 - self.de,
                                  self.x2 + self.de, self.y1 - self.de,
                                  self.x2, self.y1
                                  ], pen, brush_s1)

                    draw.polygon([self.x2 + self.de, self.y1 - self.de,
                                  self.x2, self.y1,
                                  self.x2, self.y2,
                                  self.x2 + self.de, self.y2 - self.de
                                  ], pen, brush_s2)
                    
                    # Populate projected faces for legacy mode
                    self._projected_faces = {
                        0: [(self.x1, self.y1), (self.x2, self.y1), (self.x2, self.y2), (self.x1, self.y2)], # Front
                        4: [(self.x1 + self.de, self.y1 - self.de), (self.x2 + self.de, self.y1 - self.de), (self.x2, self.y1), (self.x1, self.y1)], # Top
                        2: [(self.x2, self.y1), (self.x2 + self.de, self.y1 - self.de), (self.x2 + self.de, self.y2 - self.de), (self.x2, self.y2)] # Side (Right)
                    }

            draw.rectangle([self.x1, self.y1, self.x2, self.y2], pen, brush)
            return

        # Center of the box in 2D layout space (reference for pivot)
        cx = self.x1 + w / 2
        cy = self.y1 + h / 2
        
        # 3D vertices relative to center (x, y, z)
        # Y is Down, X is Right, Z is Out (towards viewer)
        # Vertices: 0-3 Front (z=-d/2), 4-7 Back (z=d/2)
        # Order: TL, TR, BR, BL
        dx, dy, dz = w/2, h/2, d/2
        vertices = [
            (-dx, -dy, -dz), (dx, -dy, -dz), (dx, dy, -dz), (-dx, dy, -dz), # Front
            (-dx, -dy, dz),  (dx, -dy, dz),  (dx, dy, dz),  (-dx, dy, dz)   # Back
        ]
        
        # Rotation Angles (Radians)
        theta_y = math.radians(self.rotation)
        
        # Fixed Pitch (Rotation around X) to maintain 2.5D visual style (seeing top/side)
        # If rotation is 0, we want to match the classic 'visualkeras' look which is roughly isometric/oblique.
        # Classic look: Top and Right visible.
        phi_x = math.radians(-25) # Tilt up to see top
        
        # Transform and Project
        projected = []
        for vx, vy, vz in vertices:
            # 1. Rotate Y (Yaw)
            x1 = vx * math.cos(theta_y) + vz * math.sin(theta_y)
            y1 = vy
            z1 = -vx * math.sin(theta_y) + vz * math.cos(theta_y)
            
            # 2. Rotate X (Pitch)
            x2 = x1
            y2 = y1 * math.cos(phi_x) - z1 * math.sin(phi_x)
            z2 = y1 * math.sin(phi_x) + z1 * math.cos(phi_x)
            
            # 3. Project (Orthographic + Center Offset)
            # Invert Y logic for screen coords if needed, but standard math works if we assume Y down.
            px = x2 + cx
            py = y2 + cy
            projected.append((px, py, z2))

        # Define Faces (indices)
        # Standard winding (CCW or CW). Let's define CCW looking from outside.
        faces = [
            (0, [0, 1, 2, 3], "front"),   # Front
            (1, [5, 4, 7, 6], "back"),    # Back
            (2, [1, 5, 6, 2], "right"),   # Right
            (3, [4, 0, 3, 7], "left"),    # Left
            (4, [4, 5, 1, 0], "top"),     # Top
            (5, [3, 2, 6, 7], "bottom")   # Bottom
        ]

        # Colors
        base_color = self.fill
        shade1 = fade_color(base_color, self.shade)     # Top/Bottom
        shade2 = fade_color(base_color, self.shade * 2) # Left/Right
        shade3 = fade_color(base_color, self.shade * 3) # Back/Inside
        
        face_colors = {
            "front": base_color,
            "back": shade3,
            "right": shade2,
            "left": shade2,
            "top": shade1,
            "bottom": shade1
        }

        # Calculate Face Depth (Centroid Z) for sorting
        face_depths = []
        self._projected_faces = {}
        
        for f_idx, indices, name in faces:
            # Get coords
            pts_3d = [projected[i] for i in indices]
            # Avg Z
            avg_z = sum(p[2] for p in pts_3d) / 4.0
            
            # Store 2D Quad
            quad_2d = [(p[0], p[1]) for p in pts_3d]
            self._projected_faces[f_idx] = quad_2d
            
            face_depths.append((avg_z, f_idx, indices, name, quad_2d))
            
        # Sort faces: furthest Z first (Painter's Algorithm)
        # Z increases away from camera? 
        # In our rotation math: 
        # Back (d/2) -> Rotated. 
        # Usually positive Z is towards viewer in right-hand, but here standard math
        # x_screen, y_screen. z_depth.
        # We draw from lowest Z to highest Z?
        # Let's check: Front was -d/2. Back was +d/2.
        # If we rotate 180, Front becomes +d/2.
        # We want to draw the FURTHEST face first.
        # Furthest is largest positive Z (if Z points into screen) or smallest Z (if Z points out)?
        # Our math: vertices start with Front = -d/2.
        # If Z points to viewer, -d/2 is further than +d/2? No.
        # Let's assume standard: +Z is out of screen. -Z is into screen.
        # Actually, let's just test sort. Usually standard sort (ascending) works if Z is depth.
        
        face_depths.sort(key=lambda x: x[0]) # Draw smallest Z first (furthest if Z is distance)

        # Draw
        for _, _, _, name, quad in face_depths:
            # Prepare color
            f_color = face_colors[name]
            f_pen = pen # Always black outline
            f_brush = aggdraw.Brush(f_color)
            
            # Flatten quad for aggdraw
            coords = []
            for x, y in quad:
                coords.extend([x, y])
                
            draw.polygon(coords, f_pen, f_brush)


class Circle(RectShape):

    def draw(self, draw: ImageDraw):
        pen, brush = self._get_pen_brush()
        draw.ellipse([self.x1, self.y1, self.x2, self.y2], pen, brush)


class Ellipses(RectShape):

    def draw(self, draw: ImageDraw):
        pen, brush = self._get_pen_brush()
        w = self.x2 - self.x1
        d = int(w / 7)
        draw.ellipse([self.x1 + (w - d) / 2, self.y1 + 1 * d, self.x1 + (w + d) / 2, self.y1 + 2 * d], pen, brush)
        draw.ellipse([self.x1 + (w - d) / 2, self.y1 + 3 * d, self.x1 + (w + d) / 2, self.y1 + 4 * d], pen, brush)
        draw.ellipse([self.x1 + (w - d) / 2, self.y1 + 5 * d, self.x1 + (w + d) / 2, self.y1 + 6 * d], pen, brush)


class ColorWheel:

    def __init__(self, colors: list = None):
        self._cache = dict()
        self.colors = colors if colors is not None else ["#ffd166", "#ef476f", "#06d6a0", "#118ab2", "#073b4c", "#ffadad", "#caffbf", "#9bf6ff", "#a0c4ff", "#bdb2ff"]

    def get_color(self, class_type: type):
        if class_type not in self._cache.keys():
            index = len(self._cache.keys()) % len(self.colors)
            self._cache[class_type] = self.colors[index]
        return self._cache.get(class_type)


def fade_color(color: tuple, fade_amount: int) -> tuple:
    r = max(0, color[0] - fade_amount)
    g = max(0, color[1] - fade_amount)
    b = max(0, color[2] - fade_amount)
    return r, g, b, color[3]


def get_rgba_tuple(color: Any) -> tuple:
    """

    :param color:
    :return: (R, G, B, A) tuple
    """
    if isinstance(color, tuple):
        rgba = color
    elif isinstance(color, int):
        rgba = (color >> 16 & 0xff, color >> 8 & 0xff, color & 0xff, color >> 24 & 0xff)
    else:
        rgba = ImageColor.getrgb(color)

    if len(rgba) == 3:
        rgba = (rgba[0], rgba[1], rgba[2], 255)
    return rgba


def get_keys_by_value(d, v):
    for key in d.keys():  # reverse search the dict for the value
        if d[key] == v:
            yield key


def self_multiply(tensor_tuple: tuple):
    """

    :param tensor_tuple:
    :return:
    """
    tensor_list = list(tensor_tuple)
    if None in tensor_list:
        tensor_list.remove(None)
    if len(tensor_list) == 0:
        return 0
    s = tensor_list[0]
    for i in range(1, len(tensor_list)):
        s *= tensor_list[i]
    return s


def vertical_image_concat(im1: Image, im2: Image, background_fill: Any = 'white'):
    """
    Vertical concatenation of two PIL images.

    :param im1: top image
    :param im2: bottom image
    :param background_fill: Color for the image background. Can be str or (R,G,B,A).
    :return: concatenated image
    """
    dst = Image.new('RGBA', (max(im1.width, im2.width), im1.height + im2.height), background_fill)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def linear_layout(images: list, max_width: int = -1, max_height: int = -1, horizontal: bool = True, padding: int = 0,
                  spacing: int = 0, background_fill: Any = 'white'):
    """
    Creates a linear layout of a passed list of images in horizontal or vertical orientation. The layout will wrap in x
    or y dimension if a maximum value is exceeded.

    :param images: List of PIL images
    :param max_width: Maximum width of the image. Only enforced in horizontal orientation.
    :param max_height: Maximum height of the image. Only enforced in vertical orientation.
    :param horizontal: If True, will draw images horizontally, else vertically.
    :param padding: Top, bottom, left, right border distance in pixels.
    :param spacing: Spacing in pixels between elements.
    :param background_fill: Color for the image background. Can be str or (R,G,B,A).
    :return:
    """
    coords = list()
    width = 0
    height = 0

    x, y = padding, padding

    for img in images:
        if horizontal:
            if max_width != -1 and x + img.width > max_width:
                # make a new row
                x = padding
                y = height - padding + spacing
            coords.append((x, y))

            width = max(x + img.width + padding, width)
            height = max(y + img.height + padding, height)

            x += img.width + spacing
        else:
            if max_height != -1 and y + img.height > max_height:
                # make a new column
                x = width - padding + spacing
                y = padding
            coords.append((x, y))

            width = max(x + img.width + padding, width)
            height = max(y + img.height + padding, height)

            y += img.height + spacing

    layout = Image.new('RGBA', (width, height), background_fill)
    for img, coord in zip(images, coords):
        layout.paste(img, coord)

    return layout

class Ribbon:
    def __init__(self, x1, y1, x2, y2, de, width, color, shade_step):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.de = de
        self.width = width
        self.fill = get_rgba_tuple(color)
        self.shade = shade_step
        
        # Calculate depth sort key for layering ribbons correctly
        self.z_sort = (x1 + x2) / 2 + (y1 + y2) / 2

    def draw(self, draw: aggdraw.Draw):
        pen = aggdraw.Pen("black", 0.5) # Thin outline for crispness
        
        # Colors
        top_color = fade_color(self.fill, self.shade)
        side_color = fade_color(self.fill, 2 * self.shade)
        front_color = self.fill
        
        brush_top = aggdraw.Brush(top_color)
        brush_side = aggdraw.Brush(side_color)
        brush_front = aggdraw.Brush(front_color)

        # A horizontal ribbon is a rectangle of height 'width'
        # A vertical ribbon is a rectangle of width 'width'
        
        is_horizontal = abs(self.y1 - self.y2) < abs(self.x1 - self.x2)
        
        if is_horizontal:
            # Draw Horizontal Ribbon (Left -> Right)
            lx, rx = min(self.x1, self.x2), max(self.x1, self.x2)
            y = self.y1 
            w = self.width
            
            # 1. Back Face (Top)
            # 2. Top Face (Depth)
            # Polygon: (lx, y), (rx, y), (rx+de, y-de), (lx+de, y-de)
            draw.polygon([
                lx, y - w/2, 
                rx, y - w/2, 
                rx + self.de, y - w/2 - self.de, 
                lx + self.de, y - w/2 - self.de
            ], pen, brush_top)
            
            # 3. Front Face (The main line)
            draw.rectangle([lx, y - w/2, rx, y + w/2], pen, brush_front)
            
        else:
            # Draw Vertical Ribbon (Top -> Bottom)
            ty, by = min(self.y1, self.y2), max(self.y1, self.y2)
            x = self.x1
            w = self.width
            
            # 1. Side Face
            # Polygon: (x+w/2, ty), (x+w/2, by), (x+w/2+de, by-de), (x+w/2+de, ty-de)
            draw.polygon([
                x + w/2, ty,
                x + w/2, by,
                x + w/2 + self.de, by - self.de,
                x + w/2 + self.de, ty - self.de
            ], pen, brush_side)

            # 2. Top Face
            draw.polygon([
                x - w/2, ty,
                x + w/2, ty,
                x + w/2 + self.de, ty - self.de,
                x - w/2 + self.de, ty - self.de
            ], pen, brush_top)

            # 3. Front Face
            draw.rectangle([x - w/2, ty, x + w/2, by], pen, brush_front)


def resize_image_to_fit(image: Image.Image, target_width: int, target_height: int, fit_mode: str) -> Image.Image:
    if target_width <= 0 or target_height <= 0:
        return image
        
    img_w, img_h = image.size
    target_ratio = target_width / target_height
    img_ratio = img_w / img_h
    
    new_w, new_h = target_width, target_height
    
    if fit_mode == "cover":
        if img_ratio > target_ratio:
            # Image is wider -> match height, crop width
            new_h = target_height
            new_w = int(new_h * img_ratio)
        else:
            # Image is taller -> match width, crop height
            new_w = target_width
            new_h = int(new_w / img_ratio)
    elif fit_mode == "contain":
        if img_ratio > target_ratio:
            # Image is wider -> match width, letterbox height
            new_w = target_width
            new_h = int(new_w / img_ratio)
        else:
            # Image is taller -> match height, letterbox width
            new_h = target_height
            new_w = int(new_h * img_ratio)
    elif fit_mode == "match_aspect":
        # In this mode, the container should have been resized already.
        # We just fill.
        pass
    else: # "fill"
        pass
        
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Crop or Center if needed
    if fit_mode == "cover":
        left = (new_w - target_width) // 2
        top = (new_h - target_height) // 2
        resized = resized.crop((left, top, left + target_width, top + target_height))
    elif fit_mode == "contain":
        # Create transparent background
        bg = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
        left = (target_width - new_w) // 2
        top = (target_height - new_h) // 2
        bg.paste(resized, (left, top))
        resized = bg
        
    return resized

def _calculate_affine_coeffs(quad, src_size):
    sw, sh = src_size
    p0, p1, p2, p3 = quad
    
    # Mapping:
    # p0 -> (0, 0)
    # p1 -> (sw, 0)
    # p3 -> (0, sh)
    
    # x_src = a*x_dst + b*y_dst + c
    # y_src = d*x_dst + e*y_dst + f
    
    # Matrix form for X coeffs (a, b, c):
    # [ x0 y0 1 ] [ a ]   [ 0 ]
    # [ x1 y1 1 ] [ b ] = [ sw]
    # [ x3 y3 1 ] [ c ]   [ 0 ]
    
    A = np.array([
        [p0[0], p0[1], 1],
        [p1[0], p1[1], 1],
        [p3[0], p3[1], 1]
    ])
    
    B_x = np.array([0, sw, 0])
    B_y = np.array([0, 0, sh])
    
    try:
        sol_x = np.linalg.solve(A, B_x)
        sol_y = np.linalg.solve(A, B_y)
    except np.linalg.LinAlgError:
        return (1, 0, 0, 0, 1, 0)
        
    return tuple(sol_x) + tuple(sol_y)

def apply_affine_transform(target_img: Image.Image, source_img: Image.Image, quad: list, fit_mode: str):
    """
    Maps source_img onto the quadrilateral defined by quad [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    on target_img.
    """
    # Calculate bounding box of the quad to determine target size
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = int(max_x - min_x)
    h = int(max_y - min_y)
    
    if w <= 0 or h <= 0: return

    # Let's use the side lengths.
    side_w = np.hypot(quad[1][0]-quad[0][0], quad[1][1]-quad[0][1])
    side_h = np.hypot(quad[3][0]-quad[0][0], quad[3][1]-quad[0][1])
    
    # Resize source image to match the approximate dimensions of the target face
    resized_source = resize_image_to_fit(source_img, int(side_w), int(side_h), fit_mode)
    sw, sh = resized_source.size
    
    coeffs = _calculate_affine_coeffs(quad, (sw, sh))
    
    # Transform
    # x_global = x_local + min_x
    # x_source = a*(x_local + min_x) + ...
    #          = a*x_local + b*y_local + (a*min_x + b*min_y + c)
    
    new_c = coeffs[0]*min_x + coeffs[1]*min_y + coeffs[2]
    new_f = coeffs[3]*min_x + coeffs[4]*min_y + coeffs[5]
    
    new_coeffs = (coeffs[0], coeffs[1], new_c, coeffs[3], coeffs[4], new_f)
    
    transformed = resized_source.transform((w, h), Image.AFFINE, new_coeffs, resample=Image.BILINEAR)
    
    # Paste onto target_img at (min_x, min_y)
    target_img.paste(transformed, (int(min_x), int(min_y)), transformed)


def draw_node_logo(img: Image.Image, box: Box, logo_img: Image.Image, group: Dict[str, Any], draw_volume: bool, draw_reversed: bool = False):
    axis = group.get("axis", "z")
    if not draw_volume:
        axis = "z"
    
    # Map axis to Face Index
    # z -> Front (0)
    # y -> Top (4)
    # x -> Right (2) or Left (3) depending on visibility?
    # Let's default 'x' to Right (2) as visualkeras standard.
    
    face_idx = 0
    if axis == 'y': face_idx = 4
    elif axis == 'x': face_idx = 2
    
    # Get rigorous quad from Box
    quad = box.get_face_quad(face_idx)
    if not quad or len(quad) != 4:
        return

    padding = group.get("padding", 0)
    size = group.get("size", 0.5)
    corner = group.get("corner", "top-right")

    # The quad points are corner projections: TL, TR, BR, BL (based on box vertex order 0,1,2,3)
    # However, Top face (4,5,1,0) implies (Back-TL, Back-TR, Front-TR, Front-TL).
    # We need to treat them as vectors P0..P3.
    
    p0 = np.array(quad[0])
    p1 = np.array(quad[1])
    p3 = np.array(quad[3])
    
    # Calculate vectors
    vec_x = p1 - p0
    vec_y = p3 - p0
    
    face_w = np.linalg.norm(vec_x)
    face_h = np.linalg.norm(vec_y)
    
    if face_w < 1 or face_h < 1: return

    # Normalize vectors
    u_vec_x = vec_x / face_w
    u_vec_y = vec_y / face_h
    
    # Padding vectors
    pad_vec_x = u_vec_x * padding
    pad_vec_y = u_vec_y * padding

    # Logo Sizing
    target_w, target_h = 0, 0
    if isinstance(size, (float, int)):
         scale = float(size)
         base = min(face_w, face_h)
         target_w = int(base * scale)
         if target_w <= 0: target_w = 1
         target_h = int(target_w * logo_img.height / logo_img.width)
    elif isinstance(size, (tuple, list)):
         target_w, target_h = size
    
    resized_logo = resize_image_to_fit(logo_img, target_w, target_h, "contain")
    target_w, target_h = resized_logo.size
    
    # Relativize size
    rx = target_w / face_w
    ry = target_h / face_h
    
    l_vec_x = vec_x * rx
    l_vec_y = vec_y * ry
    
    # Calculate Origin based on corner
    origin = p0 # default top-left
    
    if corner == 'top-left':
        origin = p0 + pad_vec_x + pad_vec_y
    elif corner == 'top-right':
        origin = p1 - l_vec_x - pad_vec_x + pad_vec_y
    elif corner == 'bottom-left':
        origin = p3 - l_vec_y + pad_vec_x - pad_vec_y
    elif corner == 'bottom-right':
        p2 = p0 + vec_x + vec_y
        origin = p2 - l_vec_x - l_vec_y - pad_vec_x - pad_vec_y
    elif corner == 'center':
        center = p0 + 0.5 * vec_x + 0.5 * vec_y
        origin = center - 0.5 * l_vec_x - 0.5 * l_vec_y

    # Final Logo Quad
    l_p0 = origin
    l_p1 = origin + l_vec_x
    l_p2 = origin + l_vec_x + l_vec_y
    l_p3 = origin + l_vec_y
    
    logo_quad = [tuple(l_p0), tuple(l_p1), tuple(l_p2), tuple(l_p3)]
    
    apply_affine_transform(img, resized_logo, logo_quad, "fill")


def draw_logos_legend(img: Image.Image, logo_groups: Sequence[Dict[str, Any]], legend_config: Union[bool, Dict[str, Any]], background_fill: Any, font: ImageFont.ImageFont, font_color: Any) -> Image.Image:
    if not legend_config:
        return img
        
    if isinstance(legend_config, bool):
        legend_config = {}
        
    padding = legend_config.get("padding", 10)
    spacing = legend_config.get("spacing", 10)
    
    patches = []
    
    # Determine text height for sizing
    if hasattr(font, 'getsize'):
        text_height = font.getsize("Ag")[1]
    else:
        text_height = font.getbbox("Ag")[3]
        
    # We want to show: [Logo Image] Group Name
    
    for group in logo_groups:
        name = group.get("name")
        path = group.get("file")
        if not name or not path: continue
        
        try:
            logo_img = Image.open(path)
        except:
            continue
            
        # Resize logo for legend
        # Let's make it square-ish, matching text height * 2?
        icon_size = int(text_height * 2)
        logo_img = resize_image_to_fit(logo_img, icon_size, icon_size, "contain")
        
        # Measure text
        if hasattr(font, 'getsize'):
            text_w, text_h = font.getsize(name)
        else:
            bbox = font.getbbox(name)
            text_w = bbox[2]
            text_h = bbox[3]
            
        patch_w = icon_size + spacing + text_w
        patch_h = max(icon_size, text_h)
        
        patch = Image.new("RGBA", (patch_w, patch_h), background_fill)
        draw = ImageDraw.Draw(patch)
        
        # Paste logo
        # Center vertically
        logo_y = (patch_h - icon_size) // 2
        patch.paste(logo_img, (0, logo_y), logo_img)
        
        # Draw text
        text_x = icon_size + spacing
        text_y = (patch_h - text_h) // 2
        draw.text((text_x, text_y), name, font=font, fill=font_color)
        
        patches.append(patch)
        
    if not patches:
        return img
        
    legend_image = linear_layout(patches, max_width=img.width, max_height=img.height, padding=padding,
                                 spacing=spacing,
                                 background_fill=background_fill, horizontal=True)
                                 
    return vertical_image_concat(img, legend_image, background_fill=background_fill)
