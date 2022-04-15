from scene import Scene; import taichi as ti; from taichi.math import *
scene = Scene(exposure=2); scene.set_floor(-0.05, (1.0, 1.0, 1.0)); scene.set_background_color((0, 0, 1))
n,o=128,64; J=2.0; steps=n*n*400; c1, c2 = vec3(1, 1, 0.2), vec3(0, 0.1, 0.3)
@ti.kernel
def run():
    for i, j in ti.ndrange((-o, o), (-o, o)): scene.set_voxel(vec3(i, -8, j), 1, vec3(ti.random()>0.5, 0, 0))
    ti.loop_config(serialize=True)
    for x in range(steps):
        i,j=int(ti.random()*n)-o,int(ti.random()*n)-o; s, r = scene.get_voxel(vec3(i,-8,j))[1], ti.random()
        if r < min(1, ti.exp(-2*J*(-4 + 2 * 
        ((scene.get_voxel(vec3(i,-8,(j+o+1)%n-o))[1][0]==s[0])+(scene.get_voxel(vec3(i,-8,(j+o-1)%n-o))[1][0]==s[0])
        +(scene.get_voxel(vec3((i+o+1)%n-o,-8,j))[1][0]==s[0])+(scene.get_voxel(vec3((i+o-1)%n-o,-8,j))[1][0]==s[0])))
        )): scene.set_voxel(vec3(i, -8, j), 1, vec3(1-s[0], s[1]+(r*3000>(i*i+j*j))/255, 0));\
            scene.set_voxel(vec3(i,s[1]*255%64,j),1+(r>mix(0.95,0.9,s[0])),mix(c2,c1,s[0]))
run(); scene.finish()
