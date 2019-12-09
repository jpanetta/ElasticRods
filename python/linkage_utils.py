from contextlib import redirect_stdout
import numpy as np

def printRodSegments(l, zeroBasedIndexing = False):
    rods = l.traceRods()
    offset = 0 if zeroBasedIndexing else 1
    for rod in rods:
        print('A\t' if rod[0] else 'B\t', end='')
        print('\t'.join(str(i + offset) for i in rod[1]))

def writeRodSegments(l, path = None, zeroBasedIndexing = False):
    with redirect_stdout(open(path, 'w')):
        printRodSegments(l, zeroBasedIndexing)

# Read "stiffening boxes" from a text file. Each line of this text file should represent a box
# by its 8 corner's coordinates (i.e. each line should hold 24 floating point in the order [x0, y0, z1, ...]).
# The corner numbering follows GMsh's node ordering convention:
#        v
# 3----------2
# |\     ^   |\
# | \    |   | \
# |  \   |   |  \
# |   7------+---6
# |   |  +-- |-- | -> u
# 0---+---\--1   |
#  \  |    \  \  |
#   \ |     \  \ |
#    \|      w  \|
#     4----------5
def loadStiffeningBoxes(path):
    return [np.array(list(map(float, l.strip().split()))).reshape(8, 3) for l in open(path, 'r')]


def order_segments_by_strips(linkage):
    ''' Trace segments in the linkage by following a certain orientation and 
        mark the segment as fliped if the orientation is not consistent with 
        the first segments. Compare to Tracerod, the function works for both rods with end points and ring rods. 

        The format of the return value is a list of strip, where each strip is a list of tuple, and the tuple contains the index of the segment in thsi strip and whether the segment orientation is flipped (1 for correct orientation and -1 for flipped orientation).
    '''
    def is_valid_segment_index(index):
        ''' In the rod linkage code, the NONE index is defined to be the MAX INT. To check whether a segment index is NONE, check whether the index is larger than the total number of segments. 
        '''
        return index < linkage.numSegments()

    visited_segment = set()
    total_segment = set(range(linkage.numSegments()))
    strips = []
    while len(visited_segment) < linkage.numSegments():
        seg_index = (total_segment - visited_segment).pop()
        segments_in_strip = [(seg_index, 1)]
        visited_segment.add(seg_index)
        # Trace along the startJoint -> endJoint direction.
        if linkage.segment(seg_index).endJoint != None:
            curr_index = seg_index
            next_joint = linkage.joint(linkage.segment(curr_index).endJoint)
            next_index = next_joint.continuationSegment(curr_index)
            if is_valid_segment_index(next_index):
                curr_seg_fliped = False
                while is_valid_segment_index(next_index) and (next_index not in visited_segment) and linkage.segment(curr_index).endJoint != None:
                    visited_segment.add(next_index)
                    correct_orientation_for_correct_segment = linkage.segment(next_index).startJoint == linkage.segment(curr_index).endJoint and (not curr_seg_fliped)
                    correct_orientation_for_flipped_segment = linkage.segment(next_index).startJoint == linkage.segment(curr_index).startJoint and curr_seg_fliped
                    if  correct_orientation_for_correct_segment or correct_orientation_for_flipped_segment:
                        curr_seg_fliped = False
                        segments_in_strip.append((next_index, 1))
                        next_joint = linkage.joint(linkage.segment(next_index).endJoint)
                    else:
                        curr_seg_fliped = True
                        segments_in_strip.append((next_index, -1))
                        next_joint = linkage.joint(linkage.segment(next_index).startJoint)
                    curr_index = next_index
                    next_index = next_joint.continuationSegment(next_index)
        # Trace along the endJoint -> startJoint direction
        if linkage.segment(seg_index).startJoint != None:
            curr_index = seg_index
            next_joint = linkage.joint(linkage.segment(curr_index).startJoint)
            next_index = next_joint.continuationSegment(curr_index)
            if next_index >= linkage.numSegments():
                curr_seg_fliped = False
                while is_valid_segment_index(next_index) and (next_index not in visited_segment) and linkage.segment(curr_index).startJoint != None:
                    visited_segment.add(next_index)
                    correct_orientation_for_correct_segment = linkage.segment(next_index).endJoint == linkage.segment(curr_index).startJoint and (not curr_seg_fliped)
                    correct_orientation_for_flipped_segment = linkage.segment(next_index).endJoint == linkage.segment(curr_index).endJoint and curr_seg_fliped
                    if correct_orientation_for_correct_segment or correct_orientation_for_flipped_segment:
                        curr_seg_fliped = False
                        segments_in_strip.insert(0, (next_index, 1))
                        next_joint = linkage.joint(linkage.segment(next_index).startJoint)
                    else:
                        curr_seg_fliped = True
                        segments_in_strip.insert(0, (next_index, -1))
                        next_joint = linkage.joint(linkage.segment(next_index).endJoint)
                    curr_index = next_index
                    next_index = next_joint.continuationSegment(next_index)

        strips.append(segments_in_strip)
    return strips

def get_turning_angle_and_length_from_ordered_rods(strips, linkage, rest = False):
    ''' This function takes the output of the function "order_segments_by_strips" and extract curvatures and length information. The information is organized in the same format as that function.
    '''
    all_strip_angles = []
    all_strip_edge_len = []
    all_extra_first_edge = []
    for curr_strip in strips:
        geo_curvatures = []
        edge_len = []
        # Check whether the rod is a loop
        last_segment = curr_strip[-1]
        extra_first_edge = None
        if (last_segment[1] == 1 and linkage.segment(last_segment[0]).endJoint == None) or (last_segment[1] == -1 and linkage.segment(last_segment[0]).startJoint == None):

            if rest:
                extra_first_edge = linkage.segment(curr_strip[0][0]).rod.restLengths()[0]
            else:
                extra_first_edge = linkage.segment(curr_strip[0][0]).rod.deformedConfiguration().len[0]
        all_extra_first_edge.append(extra_first_edge)

        for i in range(len(curr_strip)):
            seg_index = curr_strip[i][0]
            correct_orientation = curr_strip[i][1] == 1
            curr_curvature = None
            if rest:
                curr_curvature = np.array(linkage.segment(seg_index).rod.restKappas())[1:-1, 0]
            else:
                curr_curvature = np.array(linkage.segment(seg_index).rod.deformedConfiguration().kappa)[1:-1, 0]
            if not correct_orientation:
                curr_curvature = -1 * curr_curvature[::-1]
            geo_curvatures.extend(curr_curvature)
            curr_edge_lens = None
            # The first edge length is dropped
            if rest:
                curr_edge_lens = linkage.segment(seg_index).rod.restLengths()[1:]
            else:
                curr_edge_lens = linkage.segment(seg_index).rod.deformedConfiguration().len[1:]
            edge_len.extend(curr_edge_lens)

        geo_angles = [2 * np.arctan(k/2) for k in geo_curvatures]
        all_strip_angles.append(geo_angles)
        all_strip_edge_len.append(edge_len)
    return all_strip_angles, all_strip_edge_len, all_extra_first_edge
