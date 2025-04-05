import numpy as np
from skimage.draw import line
from skimage.morphology import thin
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw


def get_traces_data(inkml_file_abs_path, xmlns='{http://www.w3.org/2003/InkML}'):
    """
    Extracs and returns X Y coordinates from InkML file + ground truth formula in form of string.
    :param inkml_file_abs_path: path to InkML file
    :param xmlns: XML namespace
    :return: traces_data - list of dictionaries with coordinates and labels
    :return: formula - ground truth formula
    """
    traces_data = []

    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    # doc_namespace = "{http://www.w3.org/2003/InkML}"
    doc_namespace = xmlns

    'Stores traces_all with their corresponding id'
    'Remove 3rd coordinate - speed in m/s'
    traces_all = [{'id': trace_tag.get('id'),
                    'coords': [[round(float(axis_coord))
                                for axis_coord in coord[1:].split(' ')[:2]] if coord.startswith(' ')
                                else [round(float(axis_coord))
                                    for axis_coord in coord.split(' ')[:2]]
                                for coord in (trace_tag.text).replace('\n', '').split(',')]}
                    for trace_tag in root.findall(doc_namespace + 'trace')]

    'Sort traces_all list by id to make searching for references faster'
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

    'Always 1st traceGroup is a redundant wrapper'
    traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):

            label = traceGroup.find(doc_namespace + 'annotation').text

            'traces of the current traceGroup'
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):

                'Id reference to specific trace tag corresponding to currently considered label'
                traceDataRef = int(traceView.get('traceDataRef'))

                'Each trace is represented by a list of coordinates to connect'
                single_trace = traces_all[traceDataRef]['coords']
                traces_curr.append(single_trace)

            traces_data.append({'label': label, 'trace_group': traces_curr})

    else:
        'Consider Validation data that has no labels'
        'Our case - traces aren\'t grouped into traceGroups'
        # for idx, trace in enumerate(traces_all):
        #     traces_data.append({'trace_group': [trace['coords']]})
        [traces_data.append({'trace_group': [trace['coords']]})
         for trace in traces_all]
        # for idx, trace in enumerate(traces_all):
        #     traces_data.append({'label': str(idx), 'trace_group': [trace['coords']]})

    'Extract the ground truth from normalizedLabel'
    formula = None
    annotations = root.findall(doc_namespace + 'annotation')
    if annotations:
        for annotation in annotations:
            if annotation.get('type') == 'normalizedLabel':
                formula = annotation.text

    return traces_data, formula


def inkml2img(inkml_file_abs_path, img_height=int(80), line_width=int(1), padding=int(0), color="black"):
    """
    Converts InkML file to an image with specified height and line width.
    :param inkml_file_abs_path: path to InkML file
    :param img_height: height of the output image (width is variable)
    :param line_width: width of the lines in the image
    :param padding: padding around the image
    :return: image in form of np.array for easy handling (converted from  PIL Image object)
    :return: latex - ground truth formula
    """
    traces_data, formula = get_traces_data(inkml_file_abs_path)
    all_traces = []
    [all_traces.append(trace_grp['trace_group']) for trace_grp in traces_data]
    all_traces_collapsed = [trace for trace_group in all_traces for trace in trace_group]

    # Compute min and max coordinates for all traces
    min_x, min_y, max_x, max_y = get_min_coords(all_traces_collapsed)

    # Shift the trace group to align with (0, 0)
    all_traces = [shift_trace_grp(trace, min_x=min_x, min_y=min_y) for trace in all_traces]

    # Interpolate the traces to have a height of 80 pixels
    trace_height = max_y - min_y
    trace_width = max_x - min_x
    all_traces = [interpolate(trace_group, trace_grp_height=trace_height, trace_grp_width=trace_width, box_size=img_height - 1 - (padding*2)) for trace_group in all_traces]

    # Compute the new width after interpolation
    new_width = max([max([coord[0] for coord in trace]) for trace_group in all_traces for trace in trace_group]) + 1 + (padding*2)
    new_height = img_height  # Fixed height

    # Create a blank white image
    img = Image.new("RGB", (new_width, new_height), "white")
    draw = ImageDraw.Draw(img)

    # Draw the traces on the image
    for trace_group in all_traces:
        for trace in trace_group:
            if len(trace) == 1:
                # Draw a single point
                x, y = trace[0]
                draw.point((x+padding, y+padding), fill=color, size=line_width)
            else:
                # Draw lines connecting the points
                for i in range(len(trace) - 1):
                    x1, y1 = trace[i]
                    x2, y2 = trace[i + 1]
                    draw.line((x1+padding, y1+padding, x2+padding, y2+padding), fill=color, width=line_width)
    return np.array(img), formula


def get_min_coords(trace_group):
    all_coords = np.vstack(trace_group)  # Stack all traces into a single NumPy array
    min_x, min_y = np.min(all_coords, axis=0)  # Compute min for x and y
    max_x, max_y = np.max(all_coords, axis=0)  # Compute max for x and y
    return min_x, min_y, max_x, max_y


def shift_trace_grp(trace_group, min_x, min_y):
    shifted_trace_grp = [np.array(trace) - np.array([min_x, min_y]) for trace in trace_group]
    return shifted_trace_grp


def interpolate(trace_group, trace_grp_height, trace_grp_width, box_size):
    if trace_grp_height == 0:
        trace_grp_height += 1
    if trace_grp_width == 0:
        trace_grp_width += 1

    scale_factor = box_size / trace_grp_height  # Compute scale factor
    interpolated_trace_grp = [np.round(np.array(trace) * scale_factor).astype(int) for trace in trace_group]
    return interpolated_trace_grp


if __name__ == "__main__":
    import sys
    input_inkml = sys.argv[1]
    img, formula = inkml2img(input_inkml, img_height=80, line_width=1, padding=2, color='#284054')
    img.show()
    print(img.size)