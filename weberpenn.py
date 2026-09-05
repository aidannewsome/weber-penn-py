"""Weber and Penn's tree model, "Creation and Rendering of Realistic Trees" (1995), in
one Python file with no dependencies. The same as weber-penn-net, the C# port checked
against Arbaro, in structure and in every random draw, so the two grow the same tree
from the same seed. Z is up, metres.

    tree = Tree.grow(Parameters.from_paper(table), seed)
    mesh = Mesh.of(tree, sides=8)
"""
import math

MASK = (1 << 64) - 1


class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z

    def __add__(self, o):
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, s):
        return Vec3(self.x * s, self.y * s, self.z * s)

    __rmul__ = __mul__

    def __truediv__(self, s):
        return Vec3(self.x / s, self.y / s, self.z / s)

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self):
        l = self.length()
        return self / l if l > 0 else self

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o):
        return Vec3(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)

    def __repr__(self):
        return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


ZERO, UNIT_X, UNIT_Y, UNIT_Z = Vec3(), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)


def rodrigues(k, v, c, s):
    return v * c + k.cross(v) * s + k * (k.dot(v) * (1 - c))


class Frame:
    """A position and three axes; a stem grows along its Z."""
    __slots__ = ("origin", "x", "y", "z")

    def __init__(self, origin, x, y, z):
        self.origin, self.x, self.y, self.z = origin, x, y, z

    @staticmethod
    def identity():
        return Frame(Vec3(), UNIT_X, UNIT_Y, UNIT_Z)

    def apply(self, local):
        return self.origin + self.direction(local)

    def direction(self, local):
        return self.x * local.x + self.y * local.y + self.z * local.z

    def translate(self, by):
        return Frame(self.origin + by, self.x, self.y, self.z)

    def at(self, origin):
        return Frame(origin, self.x, self.y, self.z)

    def rot_x(self, degrees):
        return self.rot_local(UNIT_X, degrees)

    def rot_y(self, degrees):
        return self.rot_local(UNIT_Y, degrees)

    def rot_z(self, degrees):
        return self.rot_local(UNIT_Z, degrees)

    def rot_xz(self, down, rotate):
        """A child's direction: turned round the parent's Z, then tilted away from it."""
        return self.rot_z(rotate).rot_x(down)

    def rot_away_from_z(self, degrees, bearing):
        b = math.radians(bearing)
        return self.rot_local(Vec3(math.cos(b), math.sin(b), 0), degrees)

    def rot_local(self, axis, degrees):
        t = math.radians(degrees)
        c, s = math.cos(t), math.sin(t)
        return Frame(self.origin, self.direction(rodrigues(axis, UNIT_X, c, s)), self.direction(rodrigues(axis, UNIT_Y, c, s)), self.direction(rodrigues(axis, UNIT_Z, c, s)))

    def rot_world(self, axis, degrees):
        t = math.radians(degrees)
        c, s = math.cos(t), math.sin(t)
        k = axis.normalized()
        return Frame(self.origin, rodrigues(k, self.x, c, s), rodrigues(k, self.y, c, s), rodrigues(k, self.z, c, s))


class Rng:
    """The paper's RANDOM: SplitMix64, the same numbers as the C# for the same seed. The
    state is a plain integer, so saving and restoring it is what pruning needs."""
    __slots__ = ("state",)

    def __init__(self, seed):
        self.state = seed & MASK

    def next_raw(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def next(self):
        return (self.next_raw() >> 11) * (1.0 / 9007199254740992.0)

    def var(self, variation):
        return (self.next() * 2 - 1) * variation

    def next_seed(self):
        return self.next_raw()


class Level:
    """One level of recursion's parameters, the paper's nLength and the rest without the n."""

    def __init__(self, **kw):
        self.length = 1.0
        self.length_v = 0.0
        self.taper = 1.0
        self.seg_splits = 0.0
        self.split_angle = 0.0
        self.split_angle_v = 0.0
        self.curve_res = 1
        self.curve = 0.0
        self.curve_back = 0.0
        self.curve_v = 0.0
        self.down_angle = 0.0
        self.down_angle_v = 0.0
        self.rotate = 0.0
        self.rotate_v = 0.0
        self.branches = 0
        for k, v in kw.items():
            setattr(self, k, v)


class Parameters:
    """The paper's parameter list under the paper's names, snake cased. Level digits in
    the paper's names index into `level`; 0Scale, 0ScaleV and 0BaseSplits are
    trunk_scale, trunk_scale_v and base_splits."""

    def __init__(self, **kw):
        self.shape = 1
        self.base_size = 0.3
        self.scale = 13.0
        self.scale_v = 0.0
        self.levels = 3
        self.ratio = 0.015
        self.ratio_power = 1.2
        self.lobes = 0
        self.lobe_depth = 0.0
        self.flare = 0.0
        self.trunk_scale = 1.0
        self.trunk_scale_v = 0.0
        self.base_splits = 0
        self.level = [Level(), Level(), Level(), Level()]
        self.leaves = 0
        self.leaf_shape = 0
        self.leaf_scale = 0.17
        self.leaf_scale_x = 1.0
        self.leaf_bend = 0.0
        self.attraction_up = 0.0
        self.prune_ratio = 0.0
        self.prune_width = 0.5
        self.prune_width_peak = 0.5
        self.prune_power_low = 0.5
        self.prune_power_high = 0.5
        self.quality = 1.0
        for k, v in kw.items():
            setattr(self, k, v)

    TRUNK = {"0Scale": "trunk_scale", "0ScaleV": "trunk_scale_v", "0BaseSplits": "base_splits"}

    @staticmethod
    def from_paper(table):
        """Parameters from a table under the paper's own names, "0Length", "1DownAngle"
        and so on, as in presets.json."""
        p = Parameters()
        for key, value in table.items():
            if key in Parameters.TRUNK:
                setattr(p, Parameters.TRUNK[key], value)
            elif key[0].isdigit():
                setattr(p.level[int(key[0])], snake(key[1:]), value)
            elif key in ("ZScale", "ZScaleV"):
                continue
            else:
                setattr(p, snake(key), value)
        return p

    def level_at(self, level):
        return self.level[min(level, len(self.level) - 1)]

    def shape_ratio(self, shape, ratio):
        """The paper's ShapeRatio: a branch's length relative to the longest by its
        position up the crown, 0 at the top to 1 at the base."""
        if shape == 0:
            return 0.2 + 0.8 * ratio
        if shape == 1:
            return 0.2 + 0.8 * math.sin(math.pi * ratio)
        if shape == 2:
            return 0.2 + 0.8 * math.sin(0.5 * math.pi * ratio)
        if shape == 3:
            return 1.0
        if shape == 4:
            return 0.5 + 0.5 * ratio
        if shape == 5:
            return ratio / 0.7 if ratio <= 0.7 else (1 - ratio) / 0.3
        if shape == 6:
            return 1 - 0.8 * ratio
        if shape == 7:
            return 0.5 + 0.5 * ratio / 0.7 if ratio <= 0.7 else 0.5 + 0.5 * (1 - ratio) / 0.3
        if shape == 8:
            if ratio < 0 or ratio > 1:
                return 0.0
            if ratio < 1 - self.prune_width_peak:
                return math.pow(ratio / (1 - self.prune_width_peak), self.prune_power_high)
            return math.pow((1 - ratio) / (1 - self.prune_width_peak), self.prune_power_low)
        raise ValueError("Shape is 0 to 8.")


def snake(name):
    """LeafScaleX to leaf_scale_x."""
    return "".join("_" + ch.lower() if ch.isupper() and i > 0 else ch.lower() for i, ch in enumerate(name))


class Section:
    """A cross section of a stem: its frame, radius before lobes, and distance from the base."""
    __slots__ = ("frame", "radius", "distance")

    def __init__(self, frame, radius, distance):
        self.frame, self.radius, self.distance = frame, radius, distance

    @property
    def position(self):
        return self.frame.origin


class Leaf:
    __slots__ = ("frame", "length", "width", "shape")

    def __init__(self, frame, length, width, shape):
        self.frame, self.length, self.width, self.shape = frame, length, width, shape


class Stem:
    """The trunk or any branch: sections base to tip, children, the clones it split into,
    and leaves at the last level. A clone begins at the split and carries the sections past it."""

    def __init__(self, level, parent, offset, p):
        self.level = level
        self.parent = parent
        self.is_clone = False
        self.offset = offset
        self.length = 0.0
        self.base_radius = 0.0
        self.sections = []
        self.children = []
        self.clones = []
        self.leaves = []
        self.parameters = p

    def radius_at(self, distance):
        """The paper's radius at a distance along the stem: taper, then flare on the trunk."""
        p = self.parameters
        level = p.level_at(self.level)
        z = min(max(distance / self.length, 0.0), 1.0)
        taper = level.taper
        unit_taper = taper if taper < 1 else 2 - taper if taper < 2 else 0
        taper_z = self.base_radius * (1 - unit_taper * z)
        if taper < 1:
            radius = taper_z
        else:
            z2 = (1 - z) * self.length
            depth = 1 if taper < 2 or z2 < taper_z else taper - 2
            z3 = z2 if taper < 2 else abs(z2 - 2 * taper_z * int(z2 / (2 * taper_z) + 0.5))
            if taper < 2 and z3 >= taper_z:
                radius = taper_z
            else:
                radius = (1 - depth) * taper_z + depth * math.sqrt(max(0.0, taper_z * taper_z - (z3 - taper_z) * (z3 - taper_z)))
        if self.level == 0 and p.flare != 0:
            y = max(0.0, 1 - 8 * z)
            radius *= p.flare * (math.pow(100, y) - 1) / 100 + 1
        return radius

    def lobe_at(self, angle):
        p = self.parameters
        return 1 + p.lobe_depth * math.sin(p.lobes * angle) if self.level == 0 and p.lobes > 0 else 1.0


class Tree:
    """A tree grown from parameters and a seed: the trunk with everything on it, and the
    same stems and leaves flattened, trunk first, each stem before its clones and children."""

    def __init__(self, p, seed, scale, trunk):
        self.parameters = p
        self.seed = seed
        self.scale = scale
        self.trunk = trunk
        self.stems = []
        self.leaves = []
        self.height = 0.0
        self.width = 0.0
        if trunk is not None:
            self._collect(trunk)

    @staticmethod
    def grow(p, seed):
        return Grower.grow(p, seed)

    def _collect(self, stem):
        self.stems.append(stem)
        for s in stem.sections:
            q = s.position
            if q.z > self.height:
                self.height = q.z
            r = math.sqrt(q.x * q.x + q.y * q.y)
            if r > self.width:
                self.width = r
        self.leaves.extend(stem.leaves)
        for c in stem.clones:
            self._collect(c)
        for c in stem.children:
            self._collect(c)


class Building:
    """The state of one stem while it is being drawn; a clone starts from a copy."""

    def __init__(self, stem, lp, rng):
        self.stem = stem
        self.lp = lp
        self.seg_count = 1
        self.seg_len = 0.0
        self.split_correction = 0.0
        self.children_per_segment = 0.0
        self.leaves_per_segment = 0.0
        self.child_length_max = 0.0
        self.rot_angle = 0.0
        self.side = 1
        self.has_cloned = False
        self.prune_test = False
        self.rng = rng

    def clone_for(self, clone, seed):
        b = Building(clone, self.lp, Rng(seed))
        b.seg_count = self.seg_count
        b.seg_len = self.seg_len
        b.split_correction = self.split_correction
        b.children_per_segment = self.children_per_segment
        b.leaves_per_segment = self.leaves_per_segment
        b.child_length_max = self.child_length_max
        b.rot_angle = self.rot_angle + 180
        b.side = self.side
        b.has_cloned = True
        b.prune_test = self.prune_test
        return b


class Grower:
    """Section 4 of the paper, Tree Creation: the curved stem, splits, children, radius,
    leaves, pruning and vertical attraction, in that order."""

    def __init__(self, p, scale):
        self.p = p
        self.scale = scale
        self.trunk_scale = 1.0
        self.split_error = [0.0] * 4
        self.child_error = [0.0] * 4
        self.leaf_error = 0.0

    @staticmethod
    def grow(p, seed):
        rng = Rng(seed)
        scale = p.scale + rng.var(p.scale_v)
        grower = Grower(p, scale)
        trunk = grower.make_stem(0, Frame.identity(), None, 0.0, rng.next_seed(), 0.0)
        return Tree(p, seed, scale, trunk)

    def make_stem(self, level, frame, parent, offset, seed, parent_child_length_max):
        p = self.p
        stem = Stem(level, parent, offset, p)
        b = Building(stem, p.level_at(level), Rng(seed))
        b.seg_count = max(1, b.lp.curve_res)

        # Section 4.3: the length of the trunk, of a main branch by the shape ratio, and
        # of anything deeper by its position along its parent.
        if level == 0:
            stem.length = (b.lp.length + b.rng.var(b.lp.length_v)) * self.scale
            self.trunk_scale = p.trunk_scale + b.rng.var(p.trunk_scale_v)
        elif level == 1:
            base_length = p.base_size * self.scale
            ratio = (parent.length - offset) / (parent.length - base_length)
            stem.length = parent.length * parent_child_length_max * p.shape_ratio(p.shape, ratio)
        else:
            stem.length = parent_child_length_max * (parent.length - 0.6 * offset)
        if stem.length <= 0:
            return None
        b.seg_len = stem.length / b.seg_count

        # Section 4.4: the radius, never thicker than the parent where the child grows.
        stem.base_radius = self.base_radius(stem)
        if stem.base_radius <= 0:
            return None

        # Section 4.6: a stem that would leave the envelope is regrown shorter.
        if level > 0 and p.prune_ratio > 0:
            self.prune(b, frame)

        # Section 4.3: how many children this stem carries.
        nxt = p.level_at(level + 1)
        b.child_length_max = nxt.length + b.rng.var(nxt.length_v)
        if level < p.levels - 1:
            if level == 0:
                count = float(nxt.branches)
            elif level == 1:
                count = float(int(nxt.branches * (0.2 + 0.8 * (stem.length / parent.length) / parent_child_length_max)))
            else:
                count = float(int(nxt.branches * (1.0 - 0.5 * offset / parent.length)))
            b.children_per_segment = count / b.seg_count / (1 - p.base_size) if level == 0 else count / b.seg_count
        # Section 4.5: how many leaves, by the stem's position along its parent.
        if level == p.levels - 1 and p.leaves != 0:
            ratio = 1.0 if parent is None else offset / parent.length
            b.leaves_per_segment = abs(p.leaves) * p.shape_ratio(4, ratio) * p.quality / b.seg_count

        self.make_segments(b, 0, frame)
        return stem

    def base_radius(self, stem):
        p = self.p
        if stem.level == 0:
            return stem.length * p.ratio * self.trunk_scale
        radius = stem.parent.base_radius * math.pow(stem.length / stem.parent.length, p.ratio_power)
        return min(radius, stem.parent.radius_at(stem.offset))

    def make_segments(self, b, start, frame):
        """Draws segments start to the end, placing children, leaves and clones as it
        goes. In a pruning test, returns the first segment whose end leaves the envelope, else -1."""
        p = self.p
        stem = b.stem
        for s in range(start, b.seg_count):
            if s > 0:
                frame = self.new_direction(b, frame, s)
            d0 = s * b.seg_len
            if s == start:
                stem.sections.append(Section(frame, stem.radius_at(d0), d0))
            if not b.prune_test:
                if stem.level < p.levels - 1:
                    self.make_children(b, frame, s)
                if b.leaves_per_segment > 0:
                    self.make_leaves(b, frame, s)
            if b.lp.curve_v < 0:
                self.add_helix(b, frame, s)
            end = frame.translate(frame.z * b.seg_len)
            stem.sections.append(Section(end, stem.radius_at(d0 + b.seg_len), d0 + b.seg_len))
            frame = end
            if b.prune_test and not self.inside(end.origin):
                return s
            if s < b.seg_count - 1:
                outside, frame = self.make_clones(b, frame, s)
                if outside >= 0:
                    return outside
        return -1

    def new_direction(self, b, frame, s):
        """Section 4.1: the segment's turn, the random turn, and 4.8's pull toward the sky."""
        p = self.p
        lp = b.lp
        if lp.curve_back == 0:
            delta = lp.curve / lp.curve_res
        elif s < (lp.curve_res + 1) // 2:
            delta = lp.curve * 2 / lp.curve_res
        else:
            delta = lp.curve_back * 2 / lp.curve_res
        delta += b.split_correction
        frame = frame.rot_x(delta)
        if lp.curve_v > 0:
            frame = frame.rot_away_from_z(b.rng.var(lp.curve_v) / lp.curve_res, 180 + b.rng.var(180))
        if p.attraction_up != 0 and b.stem.level >= 2:
            declination = math.acos(max(-1.0, min(1.0, frame.z.z)))
            curve_up = p.attraction_up * declination / lp.curve_res * 180 / math.pi
            axis = Vec3(-frame.z.y, frame.z.x, 0)
            if axis.length() > 1e-9:
                frame = frame.rot_world(axis, -curve_up)
        return frame

    def add_helix(self, b, frame, s):
        """Section 4.1: a negative nCurveV makes each segment a helix, one turn per segment."""
        steps = 10
        for i in range(1, steps):
            where = i / steps
            distance = (s + where) * b.seg_len
            b.stem.sections.append(Section(frame.at(self.point_in_segment(b, frame, where)), b.stem.radius_at(distance), distance))

    def point_in_segment(self, b, frame, where):
        if b.lp.curve_v >= 0:
            return frame.apply(Vec3(0, 0, where * b.seg_len))
        lean = math.radians(abs(b.lp.curve_v))
        radius = math.tan(lean) * b.seg_len / (2 * math.pi)
        a = 2 * math.pi * where
        return frame.apply(Vec3(radius * math.cos(a) - radius, radius * math.sin(a), where * b.seg_len))

    def make_children(self, b, frame, s):
        """Section 4.3: children spaced along the segment, none on the trunk's bare base."""
        p = self.p
        stem = b.stem
        per_segment = b.children_per_segment
        offs = 0.0
        if stem.level == 0:
            base_length = p.base_size * self.scale
            seg_start = s * b.seg_len
            if seg_start + b.seg_len <= base_length:
                return
            if seg_start < base_length:
                offs = (base_length - seg_start) / b.seg_len
                per_segment *= 1 - offs
        elif s == 0:
            offs = min(1.0, stem.parent.radius_at(stem.offset) / b.seg_len)
        count = self.round(per_segment, self.child_error, min(stem.level, 3))
        if count <= 0:
            return
        dist = (1 - offs) / count
        for k in range(count):
            where = offs + dist / 2 + k * dist
            child_offset = (s + where) * b.seg_len
            direction = self.child_direction(b, frame, child_offset).at(self.point_in_segment(b, frame, where))
            child = self.make_stem(stem.level + 1, direction, stem, child_offset, b.rng.next_seed(), b.child_length_max)
            if child is not None:
                stem.children.append(child)

    def child_direction(self, b, frame, offset):
        """Section 4.3: a child's rotation round its parent and its down angle."""
        p = self.p
        stem = b.stem
        nxt = p.level_at(stem.level + 1)
        if nxt.rotate >= 0:
            b.rot_angle = math.fmod(b.rot_angle + nxt.rotate + b.rng.var(nxt.rotate_v) + 360, 360)
            rotate = b.rot_angle
        else:
            b.side = -b.side
            rotate = b.side * (180 + nxt.rotate + b.rng.var(nxt.rotate_v))
        if nxt.down_angle_v >= 0:
            down = nxt.down_angle + b.rng.var(nxt.down_angle_v)
        else:
            base_length = p.base_size * self.scale if stem.level == 0 else 0.0
            ratio = (stem.length - offset) / (stem.length - base_length)
            down = nxt.down_angle + nxt.down_angle_v * (1 - 2 * p.shape_ratio(0, ratio))
        return frame.rot_xz(down, rotate)

    def make_leaves(self, b, frame, s):
        """Section 4.5: leaves along the last level's stems, or fanned from their ends."""
        p = self.p
        stem = b.stem
        length = p.leaf_scale / math.sqrt(p.quality)
        width = p.leaf_scale * p.leaf_scale_x / math.sqrt(p.quality)
        if p.leaves > 0:
            count = self.round_leaf(b.leaves_per_segment)
            if count <= 0:
                return
            offs = min(1.0, stem.parent.radius_at(stem.offset) / b.seg_len) if s == 0 and stem.parent is not None else 0.0
            dist = (1 - offs) / count
            for k in range(count):
                where = offs + dist / 2 + k * dist
                f = self.child_direction(b, frame, (s + where) * b.seg_len).at(self.point_in_segment(b, frame, where))
                stem.leaves.append(Leaf(self.bend(f), length, width, p.leaf_shape))
        elif s == b.seg_count - 1:
            nxt = p.level_at(stem.level + 1)
            count = int(b.leaves_per_segment * b.seg_count + 0.5)
            if count <= 0:
                return
            tip = frame.translate(frame.z * b.seg_len)
            step = nxt.rotate / count
            step_v = nxt.rotate_v / count
            if count % 2 == 1:
                stem.leaves.append(Leaf(self.bend(tip), length, width, p.leaf_shape))
                first = step
            else:
                first = step / 2
            for k in range(count // 2):
                for side in (1, -1):
                    f = tip.rot_y(side * (first + k * step + b.rng.var(step_v))).rot_x(nxt.down_angle + b.rng.var(nxt.down_angle_v))
                    stem.leaves.append(Leaf(self.bend(f), length, width, p.leaf_shape))

    def bend(self, leaf):
        """Section 4.9: a leaf turned part way to face outward from the trunk, then upward."""
        p = self.p
        if p.leaf_bend == 0:
            return leaf
        pos = leaf.origin
        normal = leaf.y
        theta_position = math.atan2(pos.y, pos.x)
        theta_bend = theta_position - math.atan2(normal.y, normal.x)
        leaf = leaf.rot_world(UNIT_Z, p.leaf_bend * theta_bend * 180 / math.pi)
        normal = leaf.y
        phi_bend = math.atan2(math.sqrt(normal.x * normal.x + normal.y * normal.y), normal.z)
        return leaf.rot_x(p.leaf_bend * phi_bend * 180 / math.pi)

    def make_clones(self, b, frame, s):
        """Section 4.2: the stem forks into clones at the end of a segment. Returns the
        first segment of any clone that leaves the envelope in a pruning test, else -1,
        and the stem's own frame after the split."""
        p = self.p
        stem = b.stem
        lp = b.lp
        base_split = stem.level == 0 and s == 0 and p.base_splits > 0
        if base_split:
            splits = p.base_splits
        else:
            seg_splits = lp.seg_splits * (0.5 if stem.is_clone or b.has_cloned else 1.0)
            splits = self.round(seg_splits, self.split_error, min(stem.level, 3))
        if splits < 1:
            return -1, frame

        declination = math.acos(max(-1.0, min(1.0, frame.z.z))) * 180 / math.pi
        split_angle = max(0.0, lp.split_angle + b.rng.var(lp.split_angle_v) - declination)
        remaining = b.seg_count - s - 1
        b.split_correction -= split_angle / remaining
        b.has_cloned = True
        b.children_per_segment /= splits + 1
        b.leaves_per_segment /= splits + 1

        for i in range(1, splits + 1):
            clone = Stem(stem.level, stem.parent, stem.offset, p)
            clone.is_clone = True
            clone.length = stem.length
            clone.base_radius = stem.base_radius
            cb = b.clone_for(clone, b.rng.next_seed())
            f = frame.rot_x(split_angle)
            if base_split:
                diverge = 360.0 / (splits + 1) * i + b.rng.var(lp.split_angle_v)
            else:
                r = b.rng.next()
                diverge = 20 + 0.75 * (30 + abs(declination - 90)) * r * r
                if b.rng.next() < 0.5:
                    diverge = -diverge
            f = f.rot_world(UNIT_Z, diverge)
            outside = self.make_segments(cb, s + 1, f)
            if outside >= 0:
                return outside, frame
            stem.clones.append(clone)
        return -1, frame.rot_x(split_angle)

    def prune(self, b, frame):
        """Section 4.6: grow the stem in test, and while it leaves the envelope shorten it
        and grow it again with the same random draws."""
        p = self.p
        stem = b.stem
        original = stem.length
        saved_state = b.rng.state
        saved_correction = b.split_correction
        saved_split_error = list(self.split_error)

        def restore():
            b.rng.state = saved_state
            b.split_correction = saved_correction
            self.split_error[:] = saved_split_error
            b.has_cloned = False
            stem.sections.clear()
            stem.clones.clear()

        b.prune_test = True
        outside = self.make_segments(b, 0, frame)
        while outside >= 0 and stem.length > 0.001 * self.scale:
            restore()
            shortest = stem.length / 2
            longest = stem.length - original / 15
            stem.length = min(max(b.seg_len * outside, shortest), longest)
            b.seg_len = stem.length / b.seg_count
            stem.base_radius = self.base_radius(stem)
            outside = self.make_segments(b, 0, frame) if stem.length > 0.001 * self.scale else -1
        stem.length = original - (original - stem.length) * p.prune_ratio
        b.seg_len = stem.length / b.seg_count
        stem.base_radius = self.base_radius(stem)
        restore()
        b.prune_test = False

    def inside(self, point):
        p = self.p
        r = math.sqrt(point.x * point.x + point.y * point.y)
        ratio = (self.scale - point.z) / (self.scale * (1 - p.base_size))
        return r / self.scale < p.prune_width * p.shape_ratio(8, ratio)

    @staticmethod
    def round(value, errors, i):
        """The paper's error diffusion: a fractional count rounded so the fractions add up."""
        effective = int(math.floor(value + errors[i] + 0.5))
        errors[i] -= effective - value
        return effective

    def round_leaf(self, value):
        effective = int(math.floor(value + self.leaf_error + 0.5))
        self.leaf_error -= effective - value
        return effective


class Mesh:
    """A plain triangle mesh of a tree: every stem a tube through its sections, every leaf
    a quad. UV in metres along the bark, V along the stem and U round it by the base radius."""

    def __init__(self):
        self.vertices = []
        self.faces = []
        self.uv = []
        self.leaf_vertices = []
        self.leaf_faces = []
        self.leaf_uv = []

    @staticmethod
    def of(tree, sides=8, levels=None):
        """Up to the given sides per ring, stems below the given level only when one is
        given; a thinner stem gets fewer sides, never fewer than three."""
        mesh = Mesh()
        facet = 0.0 if tree.trunk is None else 2 * math.pi * tree.trunk.base_radius / sides
        for stem in tree.stems:
            if levels is not None and stem.level >= levels:
                continue
            n = int(math.ceil(2 * math.pi * stem.base_radius / facet)) if facet > 0 else sides
            mesh._add_stem(stem, max(3, min(sides, n)))
        for leaf in tree.leaves:
            mesh._add_leaf(leaf)
        return mesh

    def _add_stem(self, stem, sides):
        rings = []
        for i, a in enumerate(stem.sections):
            if i == 0 and stem.level == 0 and not stem.is_clone and stem.parameters.flare != 0 and len(stem.sections) > 1:
                b = stem.sections[1]
                rings.append(a)
                for k in range(6, 0, -1):
                    t = 1.0 / math.pow(2, k)
                    distance = a.distance + (b.distance - a.distance) * t
                    rings.append(Section(a.frame.at(a.position + (b.position - a.position) * t), stem.radius_at(distance), distance))
                continue
            rings.append(a)
        first = len(self.vertices)
        for ring in rings:
            for j in range(sides + 1):
                angle = 2 * math.pi * j / sides
                r = ring.radius * stem.lobe_at(angle)
                self.vertices.append(ring.frame.apply(Vec3(math.cos(angle) * r, math.sin(angle) * r, 0)))
                self.uv.append((angle * stem.base_radius, ring.distance))
        stride = sides + 1
        for i in range(len(rings) - 1):
            for j in range(sides):
                a = first + i * stride + j
                b = a + 1
                c = a + stride
                d = c + 1
                self.faces.append((a, b, d))
                self.faces.append((a, d, c))

    def _add_leaf(self, leaf):
        first = len(self.leaf_vertices)
        w = leaf.width / 2
        for local in (Vec3(-w, 0, 0), Vec3(w, 0, 0), Vec3(w, 0, leaf.length), Vec3(-w, 0, leaf.length)):
            self.leaf_vertices.append(leaf.frame.apply(local))
        self.leaf_uv += [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.leaf_faces.append((first, first + 1, first + 2))
        self.leaf_faces.append((first, first + 2, first + 3))
