import numpy as np
import elastic_rods

class Load:
    def __init__(self, p = [0, 0, 0], F = [0, 0, 0], T = [0, 0, 0]):
        self.applicationPoint = np.array(p)
        self.netForce = np.array(F)
        self.netTorque = np.array(T)

    # Changes the torque to be around a particular point
    # WARNING: discards application point of net force! This means
    # subsequent `aroundPt` calls will compute the wrong torque.
    def aroundPt(self, c):
        return Load(c, self.netForce,
                    self.netTorque + np.cross(self.applicationPoint - c, self.netForce))

    def verifyZero(self, tol=1e-7):
        fmag = np.linalg.norm(self.netForce)
        tmag = np.linalg.norm(self.netTorque)
        if (fmag > tol): raise Exception('force  imbalance: ' + str(fmag))
        if (tmag > tol): raise Exception('torque imbalance: ' + str(tmag))

    def __add__(self, b):
        if (b == 0): return self # for compatibility with sum
        if (np.any(self.applicationPoint != b.applicationPoint)):
            raise Exception('Addition of loads only implemented for common application point')
        return Load(self.applicationPoint, np.array(self.netForce) + b.netForce, np.array(self.netTorque) + b.netTorque)
    def __radd__(self, b):
        return self.__add__(b)
    def __sub__(self, b):
        return self.__add__(Load(b.applicationPoint, -b.netForce, -b.netTorque))
    def __repr__(self):
        return f'Net force: {self.netForce}\n   torque: {self.netTorque} (around pt {self.applicationPoint})'

# Get index of entities `ti` away from the terminal
def terminalVtxIndex(rod, isStart, ti):
    return ti if isStart else rod.numVertices() - 1 - ti
def terminalEdgeIndex(rod, isStart, ti):
    return ti if isStart else rod.numEdges() - 1 - ti

# Get the net force/torque on an edge according to a particular
# gradient (generalized force vector) g
def getLoadOnEdge(rod, g, edgeIdx):
    F1 = g[3 * edgeIdx:3 * (edgeIdx + 1)]
    F2 = g[3 * (edgeIdx + 1):3 * (edgeIdx + 2)]
    dc = rod.deformedConfiguration()
    edgeMidpt = np.mean(rod.deformedPoints()[edgeIdx:edgeIdx+2], axis=0)
    t = dc.tangent[edgeIdx]
    e = t * dc.len[edgeIdx]
    torqueAround = g[rod.thetaOffset() + edgeIdx]
    return Load(edgeMidpt, F1 + F2, np.cross(-e / 2, F1) + np.cross(e / 2, F2) + torqueAround * t)

# Extract "free body diagram" for a small piece of rod around a joint
# (the centerline and the applied loads); useful for setting up an equivalent
# finite element simulation on the actual joint geometry.
# keepEdges: number of edges to keep on each incident rod segment;
#            in total, `2 * keepEdges - 1` will be kept (one edge is shared)
# verificationTol: how precisely the net force/torque on the piece should vanish.
# return: (polyline, material frame vectors d1, loads)
def isolateRodPieceAtJoint(linkage, ji, ABOffset, keepEdges = 4, verificationTol = 1e-6):
    joint = linkage.joint(ji)
    segmentIdxs = joint.segments_A if ABOffset == 0 else joint.segments_B

    # Return (polyline, [load on joint edge, load on cut material interface])
    # for rod "localIdx" in [0, 1] of segment ABOffset
    def processRod(localIdx):
        si = segmentIdxs[localIdx]
        r = linkage.segment(si).rod
        isStart = joint.isStartA[localIdx] if ABOffset == 0 else joint.isStartB[localIdx]
        tei = lambda ti: terminalEdgeIndex(r, isStart, ti)
        tvi = lambda ti: terminalVtxIndex (r, isStart, ti)
        jointEdgeLoad = getLoadOnEdge(r, r.gradient(), tei(0))

        # For keepEdges=2, we keep the contributions from entities:
        # o-----o-----x--/--x
        # 0  0  1  1
        gsm = elastic_rods.GradientStencilMaskCustom()
        mask = np.zeros(r.numVertices(), dtype=bool)
        mask[0:keepEdges] = True
        if not isStart: mask = np.flip(mask)
        gsm.vtxStencilMask = mask

        mask = np.zeros(r.numEdges(), dtype=bool)
        mask[0:keepEdges] = True
        if not isStart: mask = np.flip(mask)
        gsm.edgeStencilMask = mask

        # For keepEdges=2, we get force / torque imbalances on the entities:
        #       v  v  v
        # o-----o-----x--/--x
        # 0  0  1  1
        materialInterfaceLoad = getLoadOnEdge(r, r.gradient(stencilMask=gsm), tei(keepEdges - 1))

        polylineVertices = [r.deformedPoints()[tvi(ti)] for ti in range(keepEdges + 1)] # ordered points leading away from joint
        dc = r.deformedConfiguration()
        materialFrameVectors = [dc.materialFrame[tei(ti)].d1 for ti in range(keepEdges)]
        if (localIdx == 0):
            polylineVertices = polylineVertices[::-1] # The first rod's points should lead into the joint, not away...
            materialFrameVectors = materialFrameVectors[::-1] # The first rod's points should lead into the joint...

        return (polylineVertices, materialFrameVectors, [jointEdgeLoad, materialInterfaceLoad])

    # Get the net force/torque on the rod's joint edge by summing the overlapping rod gradients
    # Also collect the loads on the two material cut interfaces afterward
    jointEdgeLoad = 0
    loads = []
    polyline = []
    materialFrameVectors = []
    ns = [joint.numSegmentsA, joint.numSegmentsB][ABOffset]
    for localRodIdx in range(2):
        if (localRodIdx >= ns): continue
        rodPolyline, rodMaterialFrameD1, rodLoads = processRod(localRodIdx)
        jointEdgeLoad += rodLoads[0]
        loads += rodLoads[1:]
        polyline += rodPolyline
        materialFrameVectors += rodMaterialFrameD1
        # print("joint edge load contribution: ", rodLoads[0])
    loads = [jointEdgeLoad] + loads

    # Delete duplicate joint edge from middle (but not if there's just one segment...)
    polyline = np.array(polyline)
    materialFrameVectors = np.array(materialFrameVectors)
    if (ns == 2):
        polyline = np.delete(polyline, [len(polyline) // 2, len(polyline) // 2 + 1], axis=0)
        materialFrameVectors = np.delete(materialFrameVectors, [len(materialFrameVectors) // 2], axis=0)

    # Validate that they match the force/torque computed from the joint variable gradient components
    g = linkage.gradient()
    jdo = linkage.dofOffsetForJoint(ji)
    # load from rod A onto joint = - load on joint edge for rod A = load on joint edge for rod B
    forceAndTorqueOnJointFromA = linkage.rivetNetForceAndTorques()[ji, :]
    forceSign = [-1.0, 1.0][ABOffset]
    jointEdgeLoadFromLinkageGradient = Load(joint.position,
                                            forceAndTorqueOnJointFromA[0:3] * forceSign,
                                            forceAndTorqueOnJointFromA[3:6] * forceSign)
    # print(jointEdgeLoadFromLinkageGradient)
    # print(jointEdgeLoad)
    (jointEdgeLoadFromLinkageGradient - jointEdgeLoad).verifyZero(verificationTol)
    # print(jointEdgeLoad.netForce, jointEdgeLoad.netTorque)

    # print("Material cut interface loads:")
    # print(loads[1].aroundPt(joint.position) + loads[2].aroundPt(joint.position))

    # Verify that all forces/torques on the segment balance
    sum([load.aroundPt(joint.position) for load in loads]).verifyZero(verificationTol)
    return (polyline, materialFrameVectors, loads)

# Get the min/max bending and twisting stress acting on the linkage
def stressesOnJointRegions(linkage, edgeDist = 3):
    nj = linkage.numJoints()
    maxBendingStresses  = np.zeros((nj, 2))
    minBendingStresses  = np.zeros((nj, 2))
    maxTwistingStresses = np.zeros((nj, 2))
    for ji in range(nj):
        j = linkage.joint(ji)
        for local_si in range(2):
            ns = [j.numSegmentsA, j.numSegmentsB][local_si]
            for si in ([j.segments_A, j.segments_B][local_si])[:ns]:
                isStart = j.terminalEdgeIdentification(si)[2]
                r = linkage.segment(si).rod
                bs = r.bendingStresses()
                ts = r.twistingStresses()
                regionStiffnesses = np.array([[bs[vi, 0], bs[vi, 1], ts[vi]] for vi in [terminalVtxIndex(r, isStart, ti) for ti in range(1, edgeDist + 1)]])
                maxBendingStresses [ji, local_si] = max(maxBendingStresses [ji, local_si], np.max(regionStiffnesses[:, 0]))
                minBendingStresses [ji, local_si] = min(minBendingStresses [ji, local_si], np.min(regionStiffnesses[:, 1]))
                maxTwistingStresses[ji, local_si] = max(maxTwistingStresses[ji, local_si], np.max(regionStiffnesses[:, 2]))
    return (maxBendingStresses, minBendingStresses, maxTwistingStresses)

def freeBodyDiagramReport(l, ji, lsi, keepEdges = 4, verificationTol=5e-6):
    centerlinePts, materialFrameD1, loads = isolateRodPieceAtJoint(l, ji, lsi, keepEdges, verificationTol=verificationTol)
    j = l.joint(ji)
    print(f"Rod segment(s) {[j.segments_A, j.segments_B][lsi][:[j.numSegmentsA, j.numSegmentsB][lsi]]} around joint {ji} at {j.position}, normal {j.normal}")
    print("Centerline points:\n", centerlinePts)

    e = np.diff(centerlinePts, axis=0)
    print("\nCenterline tangents:\n", e / np.linalg.norm(e, axis=1)[:, np.newaxis])
    print("\nCross-section frame vectors d1:\n", materialFrameD1)

    print("\nActuation torque:\t", np.abs(np.dot(loads[0].netTorque, j.normal)))
    print("Out of plane torque from rivet (shear torque):\t", np.linalg.norm(loads[0].netTorque - np.dot(loads[0].netTorque, j.normal) * j.normal))
    print(f"Load on joint edge:\n{loads[0]}")
    print(f"\nLoad on sliced material interface 1:\n{loads[1]}")
    if (len(loads) > 2):
        print(f"\nLoad on sliced material interface 2:\n{loads[2]}")
