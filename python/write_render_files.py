from matplotlib import pyplot as plt
import numpy as np, os

def writeRenderFiles(l, directory, name, stressMin = None, stressMax = None):
    polylinesA, polylinesB, points, normals, stresses = l.florinVisualizationGeometry()
    allPolylines = polylinesA[:]
    allPolylines += polylinesB[:]
    
    try: os.mkdir(directory)
    except: pass
    
    pathPrefix = directory + '/' + name
    
    polylinesFile = open(pathPrefix + '_polylines.txt', 'w')
    print("\n".join(["A " + " ".join(map(str, polyline)) for polyline in polylinesA]), file=polylinesFile)
    print("\n".join(["B " + " ".join(map(str, polyline)) for polyline in polylinesB]), file=polylinesFile)
    
    np.savetxt(pathPrefix + '_points.txt', np.array(points))
    np.savetxt(pathPrefix + '_normals.txt', np.array(normals))
    #np.savetxt(pathPrefix + '_stresses.txt', np.array(stresses))

    # write stress images
    resolution = 1000
    
    if (stressMin is None): stressMin = min(stresses);
    if (stressMax is None): stressMax = max(stresses);
    normalize = plt.Normalize(vmin=stressMin, vmax=stressMax)
    cmap = plt.cm.jet

    for idx, polyline in enumerate(allPolylines):
        nv = len(polyline)
        interpolated_stresses = np.interp(np.linspace(0, nv - 1, resolution), range(nv), [stresses[vi] for vi in polyline])
        # interpolated_stresses = np.linspace(0, stressMax, resolution)
        # interpolated_stresses[0:10] = stressMax / 2
        # interpolated_stresses[resolution - 10:] = stressMax / 2
        image = cmap(normalize(np.array(interpolated_stresses).reshape(1, resolution)))
        plt.imsave('{}_stress_{}.png'.format(pathPrefix, idx), image)

def normalized(v):
    from numpy.linalg import norm
    return v / norm(v)

def writeActuators(l, actuatedJoints, directory, name):
    outfile = open('{}/{}_actuators.txt'.format(directory, name), 'w')
    
    for ji in actuatedJoints:
        print(' '.join(map(np.format_float_scientific, np.concatenate([l.joint(ji).position, l.joint(ji).normal, normalized(l.joint(ji).e_A), normalized(l.joint(ji).e_B)]))), file=outfile)
