import os
import argparse
import pydicom
import png
import numpy as np


output_type_dict={'jpg':'+oj', 'png8':'+on', 'png16':'+on2', 'tiff':'+ot'}
output_ext={'jpg':'jpg', 'png8':'png', 'png16':'png', 'tiff':'tif'}

def dcm2png(dcm, output):
    try:
        shape = dcm.pixel_array.shape
        img_2d = dcm.pixel_array.astype(float)
        img_2d_scaled = (np.maximum(img_2d, 0) / img_2d.max()) * 255.0
        img_2d_scaled = np.uint8(img_2d_scaled)

        # save png file
        with open(output, 'wb') as png_file:
            w = png.Writer(shape[1], shape[0], greyscale=True)
            w.write(png_file, img_2d_scaled)
    except Exception as e:
        print('Failed convert jpeg2000 format, error message = {}'.format(e))
        print('output file={}'.format(output))

def dcm2img(source_path, output_path, output_type):
    file_list=os.listdir(source_path)
    for file in file_list:
        origin=os.path.join(source_path, file)
        output=os.path.join(output_path, file.replace('.dcm', '.{}'.format(output_ext[output_type])))

        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
        type_opt = output_type_dict[output_type]

        query='dcmj2pnm {} {} {}'.format(type_opt, origin, output)

        # status=subprocess_open(query)
        if os.system(query)==0:
            print('converted {} to {}'.format(origin, output))
        else:
            check_transfer_syntax(origin, output)

def check_transfer_syntax(dicom_file, output):
    dcm = pydicom.dcmread(dicom_file)
    ts_uid=dcm.file_meta.TransferSyntaxUID
    bits=dcm.BitsAllocated
    print('Transfer Syntax={}, allocated bits={}bit'.format(ts_uid, bits))
    dcm2png(dcm, output)
    print('converted {} to {}'.format(dicom_file, output))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    #parser.add_argument('--path', default='target', type=str, required=False)
    parser.add_argument('--path', default='target_jpg_20211208', type=str, required=False)
    parser.add_argument('--output_type', default='jpg', type=str, required=False)
    args=parser.parse_args()

    root_path = args.path
    source_folder_list = os.listdir(root_path)

    for folder in source_folder_list:
        dcm2img(os.path.join(root_path, folder)
                , os.path.join('{}_{}'.format(root_path, args.output_type), folder)
                , args.output_type)