import numpy as np
import elastic_rods

def localRodTime(r, t, phaseOffset):
    return t - (r - 1) * phaseOffset

def heightAtLocalTime(lt, dropDuration, dropHeight):
    return dropHeight * max(dropDuration - lt, 0.0) / dropDuration

# phase offset and drop duration in seconds
def drop_animation(directory, linkage, phaseOffset = 0.15, dropDuration = 0.5, dropHeight = None, flipZ = False, fps = 30):
    if (dropHeight == None):
        v, f, n = linkage.visualizationGeometry()
        bbSize = np.max((np.abs(np.max(v, axis=0) - np.min(v, axis=0))))
        dropHeight = 0.25 * bbSize
    if (flipZ): dropHeight *= -1

    rods = linkage.traceRods()
    numRods = len(rods)

    totalDuration = (numRods - 1) * phaseOffset + dropDuration
    totalFrameCount = int(np.ceil(30 * totalDuration))

    for frame in range(totalFrameCount):
        t = frame / fps
        frame_vertices = []
        frame_quads = []
        index_offset = 0
        for r in range(numRods):
            localTime = localRodTime(r, t, phaseOffset)
            if (localTime >= 0):
                vtxOffset = np.array([[0.0, 0.0, heightAtLocalTime(localTime, dropDuration, dropHeight)]])

                for si in rods[r][1]:
                    r_vertices, r_quads = linkage.segment(si).rod.rawVisualizationGeometry(averagedMaterialFrames = True)
                    r_vertices += vtxOffset
                    r_quads    += index_offset
                    index_offset += len(r_vertices)
                    frame_vertices.append(r_vertices)
                    frame_quads.append(r_quads)
        outVertices = np.concatenate(frame_vertices, axis=0)
        outElements = np.concatenate(frame_quads   , axis=0)

        elastic_rods.save_mesh("{}/frame_{}.msh".format(directory, frame), outVertices, outElements)
