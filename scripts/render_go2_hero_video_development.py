"""Demonstration ("hero") video of a JEPA-controlled navigation run (development; Andrew, 5 October 2026).

Presentation only: nothing here feeds an experiment. The source is one logged, successful run. The default is
`dev_prelim_off_C3_prelim_test30_ep0`: C3, the V-JEPA world model with the large past-frames decoder, on preliminary
maze 30, recovery off. It was a round trip in 136 s with 0 contacts and 14.1 cm minimum separation.

Composition (1920x1080, 30 fps, 2x simulated time):
- **Main view:** a smoothed drone-style follow camera, re-rendered from the logged trajectory (kinematic replay of the
  physics trace's base pose and 12 joint angles; no physics), in a presentation scene. The scene has the run's exact
  wall boxes with CC0 textures (assets/textures), a textured floor, soft lighting, and markers for home and the goal.
  The textures and markers are presentation choices.
- **Inset:** the authentic egocentric camera frames the controller consumed. These are the replay-verified frames from
  the preliminary video pass (`videos/prelim_C3_prelim30_recovery-off/ego_frames`, 10 Hz).
- **Minimap:** the run's wall boxes, the true trail, home and goal, and the JEPA's latest 700-ms forecast fan for its
  six candidate actions (planning.json raw_forecast_xy_m), with the selected one highlighted.
- **HUD:** phase (outbound or return, from the run's dispatch log), mission time, distance, speed, decision count, and
  the PRELIMINARY label.

Usage: render_go2_hero_video_development.py --out ~/Videos/NAME.mp4 [--source RUN] [--ego DIR] [--still SECONDS PNG]
"""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[1]
CAP = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/'
           'go2_navigation_capability_v1_attempt_001')
SOURCE = CAP/'runs/dev_prelim_off_C3_prelim_test30_ep0'
EGO = CAP/'videos/prelim_C3_prelim30_recovery-off/ego_frames'
URDF = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/lib/python3.12/'
            'site-packages/genesis/assets/urdf/go2/urdf/go2.urdf')
FLOOR_TEX = REPO/'assets/textures/floor/WoodFloor043.jpg'
WALL_TEX = REPO/'assets/textures/wall/PaintedPlaster017.jpg'
W, H, FPS, SPEED = 1920, 1080, 30, 2.0
FONT = '/usr/share/fonts/truetype/lato/Lato-{}.ttf'
ACCENT, GOAL, HOME, TEXT = (90, 200, 255), (255, 150, 40), (80, 150, 255), (240, 244, 248)


def font(weight, size):
    return ImageFont.truetype(FONT.format(weight), size)


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


class Run:
    def __init__(self, source, ego):
        self.source, self.ego = Path(source), Path(ego)
        with np.load(self.source/'native/physics_trace.npz') as z:
            self.t, self.pose, self.joints = z['timestamp_s'].copy(), z['base_pose_world'].copy(), z['joint_position'].copy()
        spec = json.loads((self.source/'specification.json').read_text())
        self.walls = spec['geometry']['wall_boxes']
        episode = json.loads((self.source/'episode.json').read_text())
        self.home, self.goal = np.asarray(episode['home_se2_world'][:2]), np.asarray(episode['beacon_xy_world'])
        self.maze = episode['maze_id']
        requests = json.loads((self.source/'requests.json').read_text())
        self.t0, self.t1 = requests[0]['now_ns']/1e9, requests[-1]['now_ns']/1e9
        self.phase = [(r['now_ns']/1e9, r.get('mission_phase')) for r in requests]
        self.decisions = []
        for r in json.loads((self.source/'planning.json').read_text()):
            mc = r.get('motion_correction')
            if mc and 'selection' in r and 'raw_forecast_xy_m' in mc:
                self.decisions.append((r['measured_ns']/1e9, np.asarray(mc['raw_forecast_xy_m']), int(r['selection']['action_index'])))
        evaluation = json.loads((self.source/'episode_evaluation.json').read_text())
        self.success = bool(evaluation['round_trip_success'])
        self.contacts = int(evaluation['disallowed_contact_samples'])
        self.clearance = float(evaluation['safety']['hard']['minimum_separation_lower_m'])
        self.ego_count = len(list(self.ego.glob('*.png')))
        xs = [w['centre_xyz'][0] for w in self.walls]
        ys = [w['centre_xyz'][1] for w in self.walls]
        self.bounds = (min(xs)-.8, max(xs)+.8, min(ys)-.8, max(ys)+.8)

    def index(self, t):
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t)-1))

    def phase_at(self, t):
        current = self.phase[0][1]
        for s, p in self.phase:
            if s > t:
                break
            current = p
        return current

    def decision_at(self, t):
        best = None
        for d in self.decisions:
            if d[0] > t:
                break
            best = d
        return best

    def ego_frame(self, t):
        k = int(np.clip((t-self.t0)/.1, 0, self.ego_count-1))
        return Image.open(self.ego/f'{k:04d}.png').convert('RGB')


def build_scene(run):
    import genesis as gs
    from lewm_genesis.rollout import DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER
    from lewm_genesis.textures import cached_box_obj
    gs.init(backend=gs.cpu, logging_level='warning')
    try:
        light = gs.options.vis.DirectionalLight(dir=(-.45, -.35, -1.), color=(1., .96, .9), intensity=3.2)
        lights = [light]
    except Exception:
        lights = None
    vis = dict(shadow=True, ambient_light=(.42, .42, .46), background_color=(.86, .9, .95), plane_reflection=False)
    if lights:
        vis['lights'] = lights
    scene = gs.Scene(show_viewer=False, vis_options=gs.options.VisOptions(**vis), renderer=gs.renderers.Rasterizer())
    x0, x1, y0, y1 = run.bounds
    size = (x1-x0+4., y1-y0+4., .04)
    texture = lambda path: gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(path), encoding='srgb'))
    scene.add_entity(gs.morphs.Mesh(file=cached_box_obj(size, tiles_per_m=.6), pos=((x0+x1)/2, (y0+y1)/2, -.02), fixed=True,
                                    collision=False, file_meshes_are_zup=True), surface=texture(FLOOR_TEX))
    for w in run.walls:
        scene.add_entity(gs.morphs.Mesh(file=cached_box_obj(tuple(w['size_xyz']), tiles_per_m=.8), pos=tuple(w['centre_xyz']),
                                        euler=(0, 0, math.degrees(w['yaw_rad'])), fixed=True, collision=False,
                                        file_meshes_are_zup=True), surface=texture(WALL_TEX))
    scene.add_entity(gs.morphs.Cylinder(radius=.035, height=.75, pos=(*run.goal, .375), fixed=True, collision=False),
                     surface=gs.surfaces.Default(color=(.95, .95, .95)))
    scene.add_entity(gs.morphs.Sphere(radius=.11, pos=(*run.goal, .82), fixed=True, collision=False),
                     surface=gs.surfaces.Emission(color=(1., .55, .12)))
    scene.add_entity(gs.morphs.Cylinder(radius=.32, height=.012, pos=(*run.goal, .006), fixed=True, collision=False),
                     surface=gs.surfaces.Default(color=(1., .6, .2)))
    scene.add_entity(gs.morphs.Cylinder(radius=.34, height=.012, pos=(*run.home, .006), fixed=True, collision=False),
                     surface=gs.surfaces.Default(color=(.3, .58, 1.)))
    robot = scene.add_entity(gs.morphs.URDF(file=str(URDF), pos=tuple(run.pose[0, :3]), fixed=False, collision=False))
    camera = scene.add_camera(res=(W, H), pos=(0., 0., 4.), lookat=(0., 0., 0.), fov=48, GUI=False, far=60.)
    scene.build()
    joints = {j.name: j for j in robot.joints}
    dofs = [int(np.asarray(joints[n].dofs_idx_local).reshape(-1)[0]) for n in DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER]
    return scene, robot, camera, dofs


class Follow:
    def __init__(self):
        self.position = self.target = self.heading = None

    def update(self, pose, alpha=.09):
        yaw = yaw_of(pose[3:])
        h = np.array([math.cos(yaw), math.sin(yaw)])
        self.heading = h if self.heading is None else (1-alpha*.6)*self.heading+alpha*.6*h
        heading = self.heading/np.linalg.norm(self.heading)
        position = np.array([*(pose[:2]-1.5*heading), 3.5])
        target = np.array([*(pose[:2]+.3*heading), .2])
        self.position = position if self.position is None else (1-alpha)*self.position+alpha*position
        self.target = target if self.target is None else (1-alpha)*self.target+alpha*target
        return self.position, self.target


def render_view(run, scene, robot, camera, dofs, follow, t):
    i = run.index(t)
    pose = run.pose[i]
    robot.set_pos(pose[:3])
    robot.set_quat([pose[6], pose[3], pose[4], pose[5]])
    robot.set_dofs_position(run.joints[i], dofs)
    position, target = follow.update(pose)
    camera.set_pose(pos=position, lookat=target, up=(0., 0., 1.))
    rgb = camera.render(rgb=True, depth=False, segmentation=False, normal=False)[0]
    return Image.fromarray(np.asarray(rgb).reshape(H, W, 3).astype(np.uint8)), i


def rounded_panel(size, radius=18, fill=(12, 16, 22, 185)):
    panel = Image.new('RGBA', size, (0, 0, 0, 0))
    ImageDraw.Draw(panel).rounded_rectangle((0, 0, size[0]-1, size[1]-1), radius=radius, fill=fill)
    return panel


def minimap(run, i, t, size=430):
    x0, x1, y0, y1 = run.bounds
    scale = (size-40)/max(x1-x0, y1-y0)
    ox, oy = 20+(size-40-(x1-x0)*scale)/2, 20+(size-40-(y1-y0)*scale)/2
    to = lambda p: (ox+(p[0]-x0)*scale, size-(oy+(p[1]-y0)*scale))
    panel = rounded_panel((size, size))
    d = ImageDraw.Draw(panel)
    for w in run.walls:
        c, s, a = np.asarray(w['centre_xyz'][:2]), np.asarray(w['size_xyz'][:2])/2, w['yaw_rad']
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        corners = [to(c+R@np.array([sx*s[0], sy*s[1]])) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        d.polygon(corners, fill=(225, 230, 238, 255))
    trail = run.pose[:i+1:50, :2]
    if len(trail) > 1:
        pts = [to(p) for p in trail]
        for k in range(1, len(pts)):
            f = k/len(pts)
            d.line([pts[k-1], pts[k]], fill=(int(60+40*f), int(150+50*f), 255, int(110+145*f)), width=4)
    for centre, colour in ((run.home, HOME), (run.goal, GOAL)):
        x, y = to(centre)
        d.ellipse((x-9, y-9, x+9, y+9), fill=(*colour, 255), outline=(255, 255, 255, 255), width=2)
    decision = run.decision_at(t)
    pose = run.pose[i]
    if decision is not None:
        td, forecast, selected = decision
        j = run.index(td)
        p0, yaw = run.pose[j, :2], yaw_of(run.pose[j, 3:])
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        for k in range(forecast.shape[0]):
            path = [to(p0)]+[to(p0+R@xy) for xy in forecast[k]]
            chosen = k == selected
            d.line(path, fill=(120, 255, 160, 255) if chosen else (255, 255, 255, 110), width=4 if chosen else 2)
    x, y = to(pose[:2])
    yaw = yaw_of(pose[3:])
    tip = (x+16*math.cos(yaw), y-16*math.sin(yaw))
    left = (x+9*math.cos(yaw+2.5), y-9*math.sin(yaw+2.5))
    right = (x+9*math.cos(yaw-2.5), y-9*math.sin(yaw-2.5))
    d.polygon([tip, left, right], fill=(*ACCENT, 255), outline=(255, 255, 255, 255))
    d.text((18, 12), 'MAP', font=font('Bold', 18), fill=(*TEXT, 230))
    legend = [((120, 255, 160), 'chosen forecast'), ((255, 255, 255), 'candidates'), (GOAL, 'goal'), (HOME, 'home')]
    lx = 72
    for colour, label in legend:
        d.rectangle((lx, 17, lx+12, 29), fill=(*colour, 255))
        d.text((lx+16, 12), label, font=font('Regular', 14), fill=(*TEXT, 210))
        lx += 26+int(d.textlength(label, font=font('Regular', 14)))
    return panel


def forecast_panel(run, t, size=(430, 300)):
    """The latest decision's six candidate 700-ms forecasts in the robot's frame (forward up), zoomed."""
    from lewm.geometry_progress_pilot_development import ACTIONS
    panel = rounded_panel(size)
    d = ImageDraw.Draw(panel)
    d.text((18, 12), 'WORLD-MODEL FORECAST  ·  next 0.7 s', font=font('Bold', 16), fill=(*TEXT, 230))
    decision = run.decision_at(t)
    origin = (size[0]//2, size[1]-46)
    scale = 1500.  # px per metre
    d.ellipse((origin[0]-7, origin[1]-7, origin[0]+7, origin[1]+7), fill=(*ACCENT, 255))
    if decision is None:
        return panel
    _, forecast, selected = decision
    names = {'forward': 'forward', 'left_arc': 'arc left', 'right_arc': 'arc right', 'left_turn': 'turn left',
             'right_turn': 'turn right', 'hold': 'hold'}
    for k in sorted(range(forecast.shape[0]), key=lambda k: k == selected):
        pts = [origin]+[(origin[0]-scale*xy[1], origin[1]-scale*xy[0]) for xy in forecast[k]]
        chosen = k == selected
        d.line(pts, fill=(120, 255, 160, 255) if chosen else (255, 255, 255, 120), width=5 if chosen else 2)
        end = pts[-1]
        d.ellipse((end[0]-4, end[1]-4, end[0]+4, end[1]+4), fill=(120, 255, 160, 255) if chosen else (255, 255, 255, 150))
    d.text((18, size[1]-30), f'chosen: {names.get(ACTIONS[selected], ACTIONS[selected])}', font=font('Bold', 18),
           fill=(120, 255, 160, 255))
    d.text((size[0]-18, size[1]-30), '10 cm', font=font('Regular', 14), fill=(*TEXT, 170), anchor='ra')
    d.line((size[0]-18-scale*.1, size[1]-12, size[0]-18, size[1]-12), fill=(*TEXT, 170), width=2)
    return panel


def compose(run, view, i, t, decisions_so_far, distance):
    frame = view.convert('RGBA')
    vignette = Image.new('L', (W, H), 0)
    ImageDraw.Draw(vignette).rectangle((0, 0, W, 170), fill=110)
    vignette = vignette.filter(ImageFilter.GaussianBlur(60))
    frame = Image.composite(Image.new('RGBA', (W, H), (8, 10, 14, 255)), frame, vignette)
    d = ImageDraw.Draw(frame)
    d.text((48, 34), 'JEPA world-model navigation', font=font('Black', 46), fill=(*TEXT, 255))
    d.text((50, 92), 'Unitree Go2 (simulated)  ·  a V-JEPA world model forecasts each candidate move from camera images  ·  reach the goal, return home',
           font=font('Regular', 22), fill=(*TEXT, 220))
    chip = rounded_panel((316, 34), radius=17, fill=(255, 196, 64, 230))
    frame.alpha_composite(chip, (50, 128))
    ImageDraw.Draw(frame).text((64, 133), 'PRELIMINARY DEV RUN · 2× SPEED', font=font('Bold', 18), fill=(30, 22, 0, 255))
    frame.alpha_composite(minimap(run, i, t), (W-430-40, 40))
    frame.alpha_composite(forecast_panel(run, t), (W-430-40, 40+430+16))
    ego = run.ego_frame(t).resize((512, 384), Image.LANCZOS)
    card = rounded_panel((540, 446))
    frame.alpha_composite(card, (40, H-446-40))
    mask = Image.new('L', ego.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, ego.size[0]-1, ego.size[1]-1), radius=12, fill=255)
    frame.paste(ego, (54, H-446-40+48), mask)
    d = ImageDraw.Draw(frame)
    d.text((56, H-446-40+12), "ROBOT CAMERA  ·  the JEPA's actual input", font=font('Bold', 18), fill=(*TEXT, 235))
    phase = run.phase_at(t)
    status = {'OUTBOUND': ('Heading to goal', GOAL), 'RETURN': ('Returning home', HOME)}.get(phase, (phase or '', ACCENT))
    if t >= run.t1-.05 and run.success:
        status = ('Home · round trip complete', (120, 255, 160))
    bar = rounded_panel((W-540-40-40-40, 112))
    bx = 540+40+40
    frame.alpha_composite(bar, (bx, H-112-40))
    d = ImageDraw.Draw(frame)
    d.ellipse((bx+26, H-112-40+44, bx+46, H-112-40+64), fill=(*status[1], 255))
    d.text((bx+58, H-112-40+32), status[0], font=font('Bold', 34), fill=(*TEXT, 255))
    a, b = max(i-250, 0), min(i+250, len(run.pose)-1)
    speed = float(np.linalg.norm(run.pose[b, :2]-run.pose[a, :2])/(run.t[b]-run.t[a]+1e-9))
    stats = [('MISSION TIME', f'{max(t-run.t0, 0.):.1f} s'), ('DISTANCE', f'{distance:.1f} m'), ('SPEED', f'{speed:.2f} m/s'),
             ('JEPA DECISIONS', f'{decisions_so_far}')]
    sx = bx+500
    for label, value in stats:
        d.text((sx, H-112-40+22), label, font=font('Bold', 14), fill=(*TEXT, 160))
        d.text((sx, H-112-40+44), value, font=font('Bold', 30), fill=(*TEXT, 255))
        sx += 185
    return frame.convert('RGB')


def end_card(frame, run):
    overlay = rounded_panel((900, 210), radius=26, fill=(10, 14, 20, 215))
    image = frame.convert('RGBA')
    image.alpha_composite(overlay, ((W-900)//2, (H-210)//2-60))
    d = ImageDraw.Draw(image)
    d.text((W//2, H//2-120), 'Round trip complete', font=font('Black', 54), fill=(120, 255, 160, 255), anchor='mm')
    d.text((W//2, H//2-52), f'{run.t1-run.t0:.0f} s  ·  {run.contacts} wall contacts  ·  closest approach {run.clearance*100:.0f} cm',
           font=font('Regular', 28), fill=(*TEXT, 240), anchor='mm')
    d.text((W//2, H//2-12), f'preliminary maze {run.maze}  ·  development run, not a sealed result', font=font('Italic', 20),
           fill=(*TEXT, 180), anchor='mm')
    return image.convert('RGB')


def main(out, source, ego, still):
    run = Run(source, ego)
    scene, robot, camera, dofs = build_scene(run)
    follow = Follow()
    if still is not None:
        for t in np.arange(run.t0, still[0], 1/FPS*SPEED):
            follow.update(run.pose[run.index(t)])
        view, i = render_view(run, scene, robot, camera, dofs, follow, still[0])
        dist = float(np.sum(np.linalg.norm(np.diff(run.pose[:i+1:25, :2], axis=0), axis=1)))
        compose(run, view, i, still[0], sum(1 for d in run.decisions if d[0] <= still[0]), dist).save(still[1])
        return
    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(['ffmpeg', '-y', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                                '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
                                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)
    times = list(np.arange(run.t0, run.t1, SPEED/FPS))+[run.t1]
    distance, last_i, last = 0., 0, None
    for t in times:
        view, i = render_view(run, scene, robot, camera, dofs, follow, t)
        distance += float(np.sum(np.linalg.norm(np.diff(run.pose[last_i:i+1:25, :2], axis=0), axis=1))) if i > last_i else 0.
        last_i = i
        last = compose(run, view, i, t, sum(1 for d in run.decisions if d[0] <= t), distance)
        encoder.stdin.write(last.tobytes())
    card = end_card(last, run)
    for _ in range(int(4*FPS)):
        encoder.stdin.write(card.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('ffmpeg failed')
    print(json.dumps(dict(out=str(out), frames=len(times)+int(4*FPS), source=str(source), success=run.success,
                          contacts=run.contacts, clearance_m=run.clearance)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='~/Videos/LeWMQuad_JEPA_navigation_hero.mp4')
    p.add_argument('--source', default=str(SOURCE))
    p.add_argument('--ego', default=str(EGO))
    p.add_argument('--still', nargs=2, metavar=('SECONDS', 'PNG'))
    a = p.parse_args()
    main(a.out, a.source, a.ego, None if a.still is None else (float(a.still[0]), a.still[1]))
