from scene import Scene; import taichi as ti; from taichi.math import *

scene = Scene(voxel_edges=0, exposure=1)
scene.set_floor(-0.05, (1.0, 1.0, 1.0))
scene.set_background_color((0, 0, 1))
scene.set_direction_light((1, 1, 1), 0.1, (1, 1, 1))
#scene.set_direction_light((1, 1, 1), 0.1, (0.01, 0.01, 0.02))
lgrid, ngrid = 15, 8

@ti.func
def is_road(i, j):
    return 0 <= i < ngrid and 0 <= j <= ngrid and scene.get_voxel(vec3(i, -8, j))[0] == 1
@ti.kernel
def initialize():
    for i, j in ti.ndrange(8, 8): scene.set_voxel(vec3(i, -8, j), 0, vec3(0))
    start, end = 1+int(vec2(ti.random(),ti.random())*(ngrid-2)), 1+int(vec2(ti.random(),ti.random())*(ngrid-2))
    turn = start + 1
    while any((abs(turn-start)==1)|(abs(turn-end)==1)): turn = 1+int(vec2(ti.random(),ti.random())*(ngrid-2))
    for k in ti.static([0, 1]):
        d = vec2(k, 1-k); p = start[k]*vec2(1-k, k)-d
        while p[1-k] < ngrid - 1:
            p += d; scene.set_voxel(vec3(p.x, -8, p.y), 1, vec3(0.5))
            if p[1-k] == turn[1-k]: d = (1 if start[k] < end[k] else -1) * vec2(1-k, k)
            if p[k] == end[k]: d = vec2(k, 1-k)
@ti.func
def build_road(X, uv, d):
    if d.sum() <= 2:
        if ((d.x | d.z) ^ (d.y | d.w)) & 1: uv = vec2(uv.y, uv.x) if (d.y | d.w) & 1 else uv
        else: # curve
            while d.z == 0 or d.w == 0: d = vec4(d.y, d.z, d.w, d.x); uv=vec2(14-uv.y, uv.x)
            uv = vec2(uv.norm(), ti.atan2(uv.x, uv.y)*2/pi*lgrid)
    elif d.sum() >= 3: # junction
        while d.sum() == 3 and d.y != 0: d = vec4(d.y, d.z, d.w, d.x); uv=vec2(14-uv.y, uv.x)  # rotate T-junction
        if d.sum() > 3 or uv.x <= 7:
            uv = vec2(mix(14-uv.x, uv.x, uv.x <= 7), mix(14-uv.y, uv.y, uv.y <= 7))
            uv = vec2(uv.norm(), ti.atan2(uv.x, uv.y)*2/pi*lgrid)
    scene.set_voxel(vec3(X.x, 0, X.y), 1, vec3(1 if uv.x==7 and 4<uv.y<12 else 0.5)) # pavement
    if uv.x <= 1 or uv.x >= 13: scene.set_voxel(vec3(X.x, 1, X.y), 1, vec3(0.7, 0.65, 0.6)) # sidewalk
    if uv.y == 7 and (uv.x == 1 or uv.x == 13): # lights
        for i in range(2, 9): scene.set_voxel(vec3(X.x, i, X.y), 1, vec3(0.6, 0.6, 0.6))
    if uv.y == 7 and (1<=uv.x<=2 or 12<=uv.x<=13): scene.set_voxel(vec3(X.x, 8, X.y), 1, vec3(0.6, 0.6, 0.6))
    if uv.y == 7 and (uv.x == 2 or uv.x == 12): scene.set_voxel(vec3(X.x, 7, X.y), 2, vec3(1, 1, 0.6))
@ti.func
def build_building(X, uv, d, r):
    while d.sum() > 0 and d.z == 0: d = vec4(d.y, d.z, d.w, d.x); uv=vec2(14-uv.y, uv.x)  # rotate
    fl = int(4 + 8 * r); style = ti.sin(r * 1234); style2 = ti.sin(r * 2345) > 0
    wall, light = vec3(mix(0.2, 0.25, style2)), mix(vec3(0.4, 0.4, 0.3), vec3(0.42, 0.45, 0.28), style2)
    for i in range(fl * 4):
        if 1 < uv.x < 13 and 1 < uv.y < 13:
            scene.set_voxel(vec3(X.x, i, X.y), mix(1, 0, i%4<2), mix(wall, light, i%4<2))
            if (uv.x == 2 or uv.x == 12) and (uv.y == 2 or uv.y == 12) or style>0 and (uv.x%3==1 or uv.y%3==1):
                scene.set_voxel(vec3(X.x, i, X.y), 1, wall)
        if 2 < uv.x < 12 and 2 < uv.y < 12:
            scene.set_voxel(vec3(X.x, i, X.y), mix(1, 2, i%4<2), mix(wall, light, i%4<2))
    scene.set_voxel(vec3(X.x, 1, X.y), 1, vec3(0.7, 0.65, 0.6)) # sidewalk
    if fl > 10 and uv.x == 6 and uv.y == 6: # antenna
        for i in range(fl+1):
            scene.set_voxel(vec3(X.x, fl*5-i, X.y), mix(1, 2, i==1), mix(vec3(0.6), vec3(0.8,0,0), i==1))
    if d.sum() > 0 and uv.y == 2 and 4 < uv.x < 10: # billboard
        for i in range(5, 7):
            scene.set_voxel(vec3(X.x,i,X.y), 2, vec3(int(r*3)==0,int(r*3)==1,int(r*3)==2)*(0.2+ti.random()*0.3))

@ti.func
def build_park(X, uv, d, r):
    h = 2 * ti.sin((uv.x**2+uv.y**2+r**2*256)/1024 * 2*pi) + 2 + (ti.random() > 0.95)
    for i in range(h):
        scene.set_voxel(vec3(X.x, i, X.y), 1, vec3(0.1, 0.4 + ti.random() * 0.2, 0.05))
@ti.kernel
def draw():
    for X in ti.grouped(ti.ndrange((-60, 60), (-60, 60))):
        I, uv = (X+60) // lgrid, float((X + 60) % lgrid)
        d = int(vec4(is_road(I.x,I.y+1),is_road(I.x+1,I.y),is_road(I.x,I.y-1),is_road(I.x-1,I.y)))
        r = mix(fract(ti.sin(dot(I, vec2(12.9898, 78.233))) * 43758.5453), any(d>0), 0.4)
        if is_road(I.x, I.y): build_road(X, uv, d)
        elif r > 0.5: build_building(X, uv, d, 2*r-1)
        else: build_park(X, uv, d, 2*r)

[initialize() for _ in range(2)]; draw(); scene.finish()
