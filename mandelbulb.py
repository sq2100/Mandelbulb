import taichi as ti
ti.init(arch=ti.cuda)
n=200
pixels = ti.Vector.field(3,dtype=float, shape=(n, n, n))
pixels_2d = ti.Vector.field(3,dtype=float, shape=(n, n))

@ti.func
def vector_power_n(v, n):
    r = v.norm()
    theta = ti.atan2(v.z, ti.sqrt(v.x ** 2 + v.y ** 2))
    phi = ti.atan2(v.y, v.x)
    return ti.Vector([ti.cos(n * phi) * ti.cos(n * theta), ti.sin(n * phi) * ti.cos(n * theta), ti.sin(n * theta)])* r**n


@ti.kernel
def paint(t:int):
    for l,m,k in pixels:
        c = ti.Vector([-0.8, 1,1])
        v = ti.Vector([3*l / n-1.5, 3*m / n-1.5 , 3*k/n-1.5])
        iterations = 0
        while v.norm() > 1 and iterations < 256:
            v = vector_power_n(v,8) + c
            iterations += 1
        pixels[l,m,k] = ti.Vector([iterations * 0.01,1,0])
        if k == t:
            pixels_2d[l,m] = ti.Vector([iterations * 0.01,1,0])



window = ti.ui.Window("Fractal 3D", (n , n), vsync=True)
canvas = window.get_canvas()
for i in range(256):
    paint(i)
    canvas.set_image(pixels_2d)
    window.show()
    filename = str(i) + 'imwrite_export.png'
    ti.imwrite(pixels_2d.to_numpy(), filename)
