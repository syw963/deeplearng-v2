import pygame
import pygame.gfxdraw
import neat
import math
import os
import sys

# ── constants ────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
FPS = 60
CAR_W, CAR_H = 20, 12
MAX_SENSOR_LEN = 300
SENSOR_ANGLES = [-90, -45, 0, 45, 90]   # relative to car heading (degrees)
NUM_CARS = 30
MAX_FRAMES_PER_GEN = 5400   # 90 sec at 60 fps → end of generation
STAGNATION_LIMIT   = 300    # frames without a new checkpoint → forced respawn

# ── colours ──────────────────────────────────────────────────────────────────
BG        = (30, 30, 40)
TRACK_COL = (70, 75, 90)
BORDER_COL= (200, 210, 230)
CAR_LIVE  = (100, 220, 120)
CAR_DEAD  = (80,  80,  90)
SENSOR_COL= (255, 200,  50, 120)
TEXT_COL  = (240, 240, 255)
CP_COL    = (80, 200, 255)

# ── track definition ─────────────────────────────────────────────────────────
# Each track boundary is a closed polygon (list of (x, y)).
# We define outer and inner walls of a circuit.

def make_track():
    """Return (outer_pts, inner_pts, checkpoints, start_pos, start_angle)."""
    cx, cy = WIDTH // 2, HEIGHT // 2

    def ellipse_pts(rx, ry, n=80, cx=cx, cy=cy):
        pts = []
        for i in range(n):
            a = 2 * math.pi * i / n
            pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
        return pts

    outer = [(100,100),(100,500),(500,500),(500,300),(1000,300),(1000,100)]
    inner = [(200,200),(200,400),(400,400),(400,210),(900,210),(900,200)]

    # Ordered gate segments spanning the track width.
    # Driving path (clockwise): top-left → down left corridor →
    #   right along bottom → up the step → right along middle →
    #   up right section → left along top → back to start.
    checkpoints = [
        ((100, 300), (200, 300)),    # 0: left corridor going DOWN
        ((300, 430), (300, 500)),    # 1: bottom corridor going RIGHT
        ((440, 350), (500, 350)),    # 2: step transition going UP
        ((700, 215), (700, 295)),    # 3: middle corridor going RIGHT
        ((900, 175), (1000, 175)),   # 4: right section going UP
        ((600, 110), (600, 195)),    # 5: top corridor going LEFT
        ((300, 110), (300, 195)),    # 6: top corridor going LEFT
    ]

    start_pos = (150, 300)
    start_angle = 180.0
    return outer, inner, checkpoints, start_pos, start_angle


# ── geometry helpers ─────────────────────────────────────────────────────────

def seg_intersect(p1, p2, p3, p4):
    """Return intersection point of segment p1-p2 and p3-p4, or None."""
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def poly_segments(pts):
    segs = []
    for i in range(len(pts)):
        segs.append((pts[i], pts[(i + 1) % len(pts)]))
    return segs


def ray_cast(origin, angle_deg, wall_segs):
    """Cast a ray; return (hit_x, hit_y, distance) to nearest wall."""
    rad = math.radians(angle_deg)
    far = (origin[0] + MAX_SENSOR_LEN * math.cos(rad),
           origin[1] + MAX_SENSOR_LEN * math.sin(rad))
    best_dist = MAX_SENSOR_LEN
    best_pt = far
    for (a, b) in wall_segs:
        pt = seg_intersect(origin, far, a, b)
        if pt:
            d = math.hypot(pt[0] - origin[0], pt[1] - origin[1])
            if d < best_dist:
                best_dist = d
                best_pt = pt
    return best_pt[0], best_pt[1], best_dist


def point_in_poly(x, y, poly):
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    px, py = x, y
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def car_on_track(cx, cy, outer_poly, inner_poly):
    return point_in_poly(cx, cy, outer_poly) and not point_in_poly(cx, cy, inner_poly)


# ── anti-aliased drawing helpers ──────────────────────────────────────────────

def draw_aa_polygon(surf, color, pts):
    """Filled AA polygon via gfxdraw."""
    ipts = [(int(x), int(y)) for x, y in pts]
    if len(ipts) >= 3:
        pygame.gfxdraw.filled_polygon(surf, ipts, color)
        pygame.gfxdraw.aapolygon(surf, ipts, color)


def draw_aa_polyline(surf, color, pts, width=3):
    """AA multi-segment line."""
    for i in range(len(pts) - 1):
        x1, y1 = int(pts[i][0]), int(pts[i][1])
        x2, y2 = int(pts[i+1][0]), int(pts[i+1][1])
        if width == 1:
            pygame.gfxdraw.line(surf, x1, y1, x2, y2, color)
        else:
            pygame.draw.line(surf, color, (x1, y1), (x2, y2), width)


def draw_track(surf, outer, inner, track_col, border_col):
    # Fill track area (outer minus inner) using surface blit trick
    # Draw filled outer polygon, then overwrite inner with BG colour
    draw_aa_polygon(surf, track_col, outer)
    draw_aa_polygon(surf, BG, inner)
    # Border lines
    closed_outer = outer + [outer[0]]
    closed_inner = inner + [inner[0]]
    draw_aa_polyline(surf, border_col, closed_outer, 3)
    draw_aa_polyline(surf, border_col, closed_inner, 3)


def draw_checkpoints(surf, checkpoints):
    for i, (a, b) in enumerate(checkpoints):
        pygame.draw.line(surf, CP_COL,
                         (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 2)
        mx, my = int((a[0] + b[0]) / 2), int((a[1] + b[1]) / 2)
        surf.blit(_cp_font.render(str(i), True, CP_COL), (mx + 3, my + 3))


def rotated_rect_pts(cx, cy, w, h, angle_deg):
    """Return 4 corners of a rotated rectangle."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    hw, hh = w / 2, h / 2
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in corners]


# ── Car class ─────────────────────────────────────────────────────────────────

class Car:
    def __init__(self, x, y, angle):
        self._sx = float(x)    # spawn x
        self._sy = float(y)    # spawn y
        self._sa = float(angle)
        self.x = float(x)
        self.y = float(y)
        self.angle = float(angle)   # degrees; 0 = right, 90 = down
        self.speed = 3.0
        self.fitness = 0.0
        self.frames = 0
        self.cp_count      = 0   # total checkpoints crossed (cumulative)
        self.next_cp       = 0   # index of next checkpoint to collect
        self.respawns      = 0
        self.frames_since_cp = 0  # stagnation counter
        self._prev    = (float(x), float(y))
        self.sensor_hits = [(x, y)] * 5
        self.sensor_dists = [MAX_SENSOR_LEN] * 5

    def _respawn(self):
        self.x, self.y, self.angle = self._sx, self._sy, self._sa
        self.speed = 3.0
        self.respawns += 1
        self.frames_since_cp = 0
        self._prev = (self._sx, self._sy)
        self.sensor_hits  = [(self._sx, self._sy)] * 5
        self.sensor_dists = [MAX_SENSOR_LEN] * 5

    def update(self, net, wall_segs, outer_poly, inner_poly, checkpoints):
        self._prev = (self.x, self.y)

        # ── neural-net inference ──────────────────────────────────────────
        inputs = [d / MAX_SENSOR_LEN for d in self.sensor_dists]
        steer, accel = net.activate(inputs)

        # steer ∈ (-1,1) tanh → max ±4 deg/frame
        self.angle += steer * 4.0
        # accel ∈ (-1,1) → speed 1..6
        self.speed = max(1.0, min(6.0, self.speed + accel * 0.5))

        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)
        self.frames += 1

        # ── checkpoint detection ──────────────────────────────────────────
        cp_a, cp_b = checkpoints[self.next_cp]
        if seg_intersect(self._prev, (self.x, self.y), cp_a, cp_b):
            self.cp_count += 1
            self.next_cp = (self.next_cp + 1) % len(checkpoints)
            self.frames_since_cp = 0
        else:
            self.frames_since_cp += 1

        # checkpoints dominate; frames are a small tiebreaker only
        self.fitness = self.cp_count * 500 + self.frames * 0.01

        # ── stagnation → respawn ─────────────────────────────────────────
        if self.frames_since_cp >= STAGNATION_LIMIT:
            self._respawn()
            return

        # ── collision → respawn ──────────────────────────────────────────
        if not car_on_track(self.x, self.y, outer_poly, inner_poly):
            self._respawn()
            return

        # ── sensors ──────────────────────────────────────────────────────
        for i, rel_angle in enumerate(SENSOR_ANGLES):
            abs_angle = self.angle + rel_angle
            hx, hy, dist = ray_cast((self.x, self.y), abs_angle, wall_segs)
            self.sensor_hits[i] = (hx, hy)
            self.sensor_dists[i] = dist

    def draw(self, surf):
        pts = rotated_rect_pts(self.x, self.y, CAR_W, CAR_H, self.angle)
        draw_aa_polygon(surf, CAR_LIVE, pts)

        sensor_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for hit in self.sensor_hits:
            pygame.draw.line(sensor_surf, SENSOR_COL,
                             (int(self.x), int(self.y)),
                             (int(hit[0]), int(hit[1])), 1)
        surf.blit(sensor_surf, (0, 0))


# ── NEAT eval function ────────────────────────────────────────────────────────

def run_simulation(genomes, config):
    global generation, screen, clock, font_lg, font_sm
    generation += 1

    outer, inner, checkpoints, start_pos, start_angle = make_track()
    all_segs = poly_segments(outer) + poly_segments(inner)

    nets, cars = [], []
    for _, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car(*start_pos, start_angle))

    # initial sensor pass
    for car in cars:
        for i, rel in enumerate(SENSOR_ANGLES):
            hx, hy, dist = ray_cast((car.x, car.y), car.angle + rel, all_segs)
            car.sensor_hits[i] = (hx, hy)
            car.sensor_dists[i] = dist

    gen_frame = 0
    while gen_frame < MAX_FRAMES_PER_GEN:
        clock.tick(FPS)
        gen_frame += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()

        # ── update ───────────────────────────────────────────────────────
        for net, car, (_, genome) in zip(nets, cars, genomes):
            car.update(net, all_segs, outer, inner, checkpoints)
            genome.fitness = car.fitness

        best_cp      = max(c.cp_count  for c in cars)
        total_spawns = sum(c.respawns  for c in cars)

        # ── draw ─────────────────────────────────────────────────────────
        screen.fill(BG)
        draw_track(screen, outer, inner, TRACK_COL, BORDER_COL)
        draw_checkpoints(screen, checkpoints)

        for car in cars:
            car.draw(screen)

        # HUD
        screen.blit(font_lg.render(f"Generation: {generation}", True, TEXT_COL), (18, 14))
        screen.blit(font_sm.render(f"Respawns: {total_spawns}", True, TEXT_COL), (18, 52))
        screen.blit(font_sm.render(f"Best CP: {best_cp}", True, CP_COL), (18, 76))

        pygame.display.flip()


# ── entry point ───────────────────────────────────────────────────────────────

generation = 0
_cp_font = None

def main():
    global screen, clock, font_lg, font_sm, _cp_font

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT Self-Driving Car Evolution")
    clock = pygame.time.Clock()
    font_lg  = pygame.font.SysFont("Arial", 28, bold=True)
    font_sm  = pygame.font.SysFont("Arial", 20)
    _cp_font = pygame.font.SysFont("Arial", 13)

    config_path = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())

    pop.run(run_simulation, 1000)


if __name__ == "__main__":
    main()
