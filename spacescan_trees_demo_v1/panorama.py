"""Equirectangular sphere -> pinhole views. Geometry only, no learned enhancement.

Coordinate convention: panorama x=.5 is yaw=0; y=.5 is the horizon.
Yaw increases to the right; pitch increases upward. No physical height inferred.
"""
import math
import numpy as np
from PIL import Image


def resolve_projection(size, requested='auto'):
    if requested not in {'auto', 'photo', 'equirectangular'}:
        raise ValueError('Invalid projection / 影像投影類型無效')
    ratio = size[0]/size[1]
    if requested == 'equirectangular' and not 1.96 <= ratio <= 2.04:
        raise ValueError('Use an uncropped 2:1 spherical panorama. / 請使用未裁剪、比例 2:1 的球面全景。')
    is_pano = requested == 'equirectangular' or (requested == 'auto' and 1.96 <= ratio <= 2.04)
    return {'kind': 'equirectangular' if is_pano else 'photo',
            'inferred_from_ratio': requested == 'auto' and is_pano,
            'requested': requested}


def basis(yaw, pitch):
    a, b = np.radians([yaw, pitch])
    forward = np.array([math.cos(b)*math.sin(a), math.sin(b), math.cos(b)*math.cos(a)])
    right = np.array([math.cos(a), 0, -math.sin(a)])
    up = np.cross(forward, right)
    return forward, right, up


def uv_rays(u, v, spec):
    forward, right, up = basis(spec['yaw'], spec['pitch'])
    tx, ty = np.tan(np.radians([spec['hfov'], spec['vfov']])/2)
    return forward + np.asarray(2*u-1)[...,None]*tx*right + np.asarray(1-2*v)[...,None]*ty*up


def rays_uv(rays, reference_yaw=0):
    longitude = np.arctan2(rays[...,0], rays[...,2])
    reference = math.radians(reference_yaw)
    longitude = reference + (longitude-reference+np.pi)%(2*np.pi)-np.pi
    latitude = np.arctan2(rays[...,1], np.hypot(rays[...,0], rays[...,2]))
    return longitude/(2*np.pi)+.5, .5-latitude/np.pi


def view_spec(size, yaw, pitch, hfov=90, vfov=110, max_edge=3072):
    # Output focal length at center <= native panorama angular sampling density.
    focal = size[0]/(2*np.pi)
    w, h = 2*focal*np.tan(np.radians([hfov,vfov])/2)
    scale = min(1, max_edge/max(w,h))
    return {'yaw':float(yaw), 'pitch':float(pitch), 'hfov':float(hfov), 'vfov':float(vfov),
            'width':max(32,round(w*scale)), 'height':max(32,round(h*scale))}


def render(image, spec, max_edge=None):
    import cv2
    w,h = spec['width'],spec['height']
    if max_edge:
        scale=min(1,max_edge/max(w,h));w,h=max(1,round(w*scale)),max(1,round(h*scale))
    x=(np.arange(w,dtype=np.float32)+.5)/w
    y=(np.arange(h,dtype=np.float32)+.5)/h
    u,v=np.meshgrid(x,y)
    px,py=rays_uv(uv_rays(u,v,spec),spec['yaw'])
    # Horizontal wrap only; vertical poles are clamped, not wrapped to the other pole.
    mx=((px%1)*image.width-.5).astype(np.float32)
    my=np.clip(py*image.height-.5,0,image.height-1).astype(np.float32)
    src=np.asarray(image)
    # Pad one column on each side for bilinear interpolation across the seam.
    src=np.concatenate([src[:,-1:],src,src[:,:1]],axis=1)
    return Image.fromarray(cv2.remap(src,mx+1,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE))


def detection_views(size):
    # Overlapping views of roadside trees above/below the camera; not a complete
    # spherical instance census. The extreme poles are not a reliable tree view.
    return [view_spec(size,yaw,pitch) for pitch in (30,-30) for yaw in range(-180,180,45)]


def sphere_box(box, spec):
    a,b,c,d=box;t=np.linspace(0,1,17)
    u=np.concatenate([a+(c-a)*t,a+(c-a)*t,np.full(17,a),np.full(17,c)])
    v=np.concatenate([np.full(17,b),np.full(17,d),b+(d-b)*t,b+(d-b)*t])
    x,y=rays_uv(uv_rays(u,v,spec),spec['yaw'])
    return [float(x.min()),float(y.min()),float(x.max()),float(y.max())]


def perspective_for_region(image, region, max_edge=3072):
    """Controlled audit: fit the same spherical region in one perspective view.

    Fits its sampled perimeter and masks pixels outside the old panorama region,
    so the comparison cannot silently gain a neighboring tree or lose old content.
    """
    a,b,c,d=region
    yaw=((a+c)/2-.5)*360;pitch=(.5-(b+d)/2)*180
    t=np.linspace(0,1,101)
    u=np.concatenate([a+(c-a)*t,a+(c-a)*t,np.full(101,a),np.full(101,c)])
    v=np.concatenate([np.full(101,b),np.full(101,d),b+(d-b)*t,b+(d-b)*t])
    lon=(u-.5)*2*np.pi;lat=(.5-v)*np.pi
    rays=np.stack([np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)],axis=1)
    f,r,up=basis(yaw,pitch);depth=rays@f
    if depth.min()<=.05:
        raise ValueError('Region too wide for one pinhole view')
    tx=np.max(np.abs((rays@r)/depth));ty=np.max(np.abs((rays@up)/depth))
    spec=view_spec(image.size,yaw,pitch,math.degrees(2*math.atan(tx)),math.degrees(2*math.atan(ty)),max_edge)
    out=render(image,spec)
    x,y=np.meshgrid((np.arange(out.width)+.5)/out.width,(np.arange(out.height)+.5)/out.height)
    pu,pv=rays_uv(uv_rays(x,y,spec),yaw)
    pixels=np.asarray(out).copy();pixels[(pu<a)|(pu>c)|(pv<b)|(pv>d)]=128
    return Image.fromarray(pixels),spec


def circular_iou(a,b):
    def area(x):return max(0,x[2]-x[0])*max(0,x[3]-x[1])
    best=0
    for shift in (-1,0,1):
        moved=[b[0]+shift,b[1],b[2]+shift,b[3]]
        inter=max(0,min(a[2],moved[2])-max(a[0],moved[0]))*max(0,min(a[3],moved[3])-max(a[1],moved[1]))
        best=max(best,inter/max(1e-9,area(a)+area(moved)-inter))
    return best
