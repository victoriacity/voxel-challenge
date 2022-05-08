from scene import Scene
from renderer import Renderer
import taichi as ti
from taichi.math import *

lgrid = 15; road_density = 2; l = 8

@ti.func
def rand(i, j): return fract(ti.sin(dot(vec2(i, j), vec2(12.9898, 78.233))) * 43758.5453)

@ti.func
def is_road(x, y):
    mask = 0
    cx = int(rand(x // l, y // l) * l)
    cy = int(rand(x // l, y // l + l) * l)
    for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
        if 0 < abs(dx) + abs(dy) < 2:
            new_cx = int((rand(x // l + dx, y // l + dy) + dx) * l)
            new_cy = int((rand(x // l + dx, y // l + dy + l) + dy) * l)
            corner = rand(2 * (x // l) + dx, 2 * (y // l) + dy) > 0.5
            edge1 = (x % l == min(cx, new_cx)) & (min(cy, new_cy) <= y % l) & (y % l <= max(cy, new_cy))
            edge2 = (y % l == min(cy, new_cy)) & (min(cx, new_cx) <= x % l) & (x % l <= max(cx, new_cx))
            edge3 = (x % l == max(cx, new_cx)) & (min(cy, new_cy) <= y % l) & (y % l <= max(cy, new_cy))
            edge4 = (y % l == max(cy, new_cy)) & (min(cx, new_cx) <= x % l) & (x % l <= max(cx, new_cx))
            path1 = edge1 | ti.select((cx - new_cx) * (cy - new_cy) < 0, edge2, edge4)
            path2 = edge3 | ti.select((cx - new_cx) * (cy - new_cy) < 0, edge4, edge2)
            mask += ti.select(corner, path1, path2)
    return mask > 0

@ti.func
def build_road(X, uv, d):
    mat, color = 0, vec3(0)
    if X.y < 9:
        if d.sum() <= 2:
            if ((d.x | d.z) ^ (d.y | d.w)) & 1: uv = vec2(uv.y, uv.x) if (d.y | d.w) & 1 else uv
            else: # curve
                while d.z == 0 or d.w == 0: d = vec4(d.y, d.z, d.w, d.x); uv=vec2(14-uv.y, uv.x)
                uv = vec2(uv.norm(), ti.atan2(uv.x, uv.y)*2/pi*lgrid)
        elif d.sum() >= 3: # junction
            while d.sum() == 3 and d.y != 0: d = vec4(d.y, d.z, d.w, d.x); uv=vec2(14-uv.y, uv.x) # rotate T-junction
            if d.sum() > 3 or uv.x <= 7:
                uv = vec2(mix(14-uv.x, uv.x, uv.x <= 7), mix(14-uv.y, uv.y, uv.y <= 7))
                uv = vec2(uv.norm(), ti.atan2(uv.x, uv.y)*2/pi*lgrid)
        if X.y == 0:
            mat, color = 1, vec3(1 if uv.x==7 and 4<uv.y<12 else 0.5) # pavement
        elif X.y == 1:
            if uv.x <= 1 or uv.x >= 13: mat, color = 1, vec3(0.7, 0.65, 0.6) # sidewalk
        else:
            if uv.y == 7 and (uv.x == 1 or uv.x == 13): # lights
                mat, color = 1, vec3(0.6, 0.6, 0.6)
            if X.y == 8 and uv.y == 7 and (1<=uv.x<=2 or 12<=uv.x<=13): mat, color = 1, vec3(0.6, 0.6, 0.6)
            if X.y == 7 and uv.y == 7 and (uv.x == 2 or uv.x == 12): mat, color = 2, vec3(1, 1, 0.6)
    return mat, color

@ti.func
def build_building(X, uv, d, r):
    mat, color = 0, vec3(0)
    while d.sum() > 0 and d.z == 0: d = vec4(d.y, d.z, d.w, d.x); uv=vec2(14-uv.y, uv.x)  # rotate
    fl = int(3 + 20 * (r ** 12)); style = rand(r, 5)
    wall = vec3(rand(r, 1),rand(r, 2),rand(r, 2)) * 0.2+0.4
    wall2 = mix(vec3(rand(r, 9)*0.2+0.2), wall, style > 0.5 and rand(r, 4) < 0.4)
    maxdist = max(abs(uv.x - 7), abs(uv.y - 7))
    if 2 <= X.y < fl * 4:
        light = mix(vec3(0.25,0.35,0.38), vec3(0.7,0.7,0.6), rand(rand(X.x, X.y), X.y//2)>0.6)
        if maxdist < 6:
            mat, color =  mix(1, 0, X.y%4<2), mix(wall2, light, X.y%4<2)
            if (uv.x == 2 or uv.x == 12) and (uv.y == 2 or uv.y == 12) or style>0.5 and (uv.x%3==1 or uv.y%3==1):
                mat, color = 1, wall
        if maxdist < 5:  mat, color = mix(1, 2, X.y%4<2), mix(wall, light, X.y%4<2)
    if fl * 4 <= X.y < fl * 4 + 2 and maxdist == 5: 
        mat, color = 1, wall # roof
    if X.y == fl * 4 and maxdist < 5: mat, color = 1, vec3(rand(r, 7)*0.2+0.4)
    if X.y < 2: mat, color =  1, vec3(0.7, 0.65, 0.6) # sidewalk
    if fl > 15 and uv.x == 6 and uv.y == 6: # antenna
        if X.y == fl * 5: mat, color = 2, vec3(0.8, 0, 0)
        if fl*4 <= X.y < fl * 5: mat, color = 1, vec3(0.6)
    if d.sum() > 0 and uv.y == 2 and 4 < uv.x < 10: # billboard
        if 5 <= X.y < 7:
            mat, color = 2, vec3(int(r*3)==0,int(r*3)==1,int(r*3)==2)*(0.2+ti.random()*0.3)
        elif 2 <= X.y < 5:
            mat, color = 0, vec3(0)
    if d.sum() > 0 and uv.y == 3 and 4 < uv.x < 10:
        if 2 <= X.y < 5:
            mat, color = 1, vec3(0.7,0.7,0.6)
    if max(abs(uv.x - rand(r, 8)*7-4), abs(uv.y - rand(r, 10)*7-4)) < 1.5: # HVAC
        if fl*4+1 < X.y < fl*4+3: mat, color = 1, vec3(0.6)
    return mat, color

@ti.func
def build_park(X, uv, d, r):
    mat, color = 0, vec3(0)
    center, height = int(vec2(rand(r, 1) * 7 + 4, rand(r, 2) * 7 + 4)), 9 + rand(r, 3) * 5
    if X.y < height + 3: # tree
        if (uv - center).norm() < 1:
            mat, color = 1, vec3(0.36, 0.18, 0.06)
        if X.y > min(height-4, (height+5)//2) and (uv - center).norm() < (height+3-X.y) * (rand(r, 4)*0.6 + 0.4):
            mat, color = ti.random()<0.8, vec3(0.1, 0.3 + ti.random()*0.2, 0.1)
    h = 2 * ti.sin((uv.x**2+uv.y**2+rand(r, 0)**2*256)/1024 * 2*pi) + 2 + (ti.random() > 0.95)
    if X.y < h: # grass
        mat, color = 1, vec3(0.2, 0.5 + ti.random() * 0.2, 0.05)
    if max(abs(uv.x - rand(r, 4)*7-4), abs(uv.y - rand(r, 5)*7-4)) < 0.5: # light
        if h <= X.y < h + 3:
            mat, color = 1+(X.y==h+1), mix(vec3(0.2),vec3(0.9,0.8,0.6),vec3(X.y==h+1))
    return mat, color


class CityRenderer(Renderer):
    @ti.func
    def get_voxel(self, X):
        mat, color = 1, vec3(0.5)
        I, uv = int(vec2(X.x, X.z) // lgrid), float(vec2(X.x, X.z) % lgrid)
        d = int(vec4(is_road(I.x,I.y+1),is_road(I.x+1,I.y),is_road(I.x,I.y-1),is_road(I.x-1,I.y)))
        if X.y >= 0:
            r = mix(rand(I.x, I.y), any(d>0), 0.4)
            if is_road(I.x, I.y) and d.sum() > 1:
                mat, color = build_road(X, uv, d)   
            elif r > 0.5:
                mat, color = build_building(X, uv, d, 2*r-1)
            else:
                mat, color = build_park(X, uv, d, 2*r)
        return mat, color
    
day = False; manual_seed = 77
scene = Scene(CityRenderer, voxel_edges=0, exposure=2.5 - day)
scene.set_floor(-0.05, (1.0, 1.0, 1.0))
scene.set_background_color((0.9, 0.98, 1) if day else (0.01, 0.01, 0.02))
scene.set_directional_light((1, 1, -1), 0.1, (0.9, 0.98, 1) if day else (0.01, 0.01, 0.02))
scene.finish()