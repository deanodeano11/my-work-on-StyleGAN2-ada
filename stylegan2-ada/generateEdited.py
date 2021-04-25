# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import argparse
import os
import pickle
import re

import tensorflow as tf
import numpy as np
import PIL.Image
import json

import dnnlib
import dnnlib.tflib as tflib

#----------------------------------------------------------------------------
#Custom method from me
def write_json(target_path, target_file, data):
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)
#----------------------------------------------------------------------------
#pass in the noise_vars values and z values
def generate_images(network_pkl, truncation_psi, outdir, class_idx, importedNoiseVars, importedZVariables): #dlatents is a numpy representation of an array of latent variables(vector with a certain number of dimensions)
    tflib.init_tf() #initialise tensorflow
    print('Loading networks from "%s"...' % network_pkl)

    with dnnlib.util.open_url(network_pkl) as fp: #open the path of the pkl file
        _G, _D, Gs = pickle.load(fp) # load up the pkl file

    os.makedirs(outdir, exist_ok=True) # make output directory

    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False #dont use randomised noise
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi



    label = np.zeros([1] + Gs.input_shapes[1][1:])
    if class_idx is not None:
        label[:, class_idx] = 1

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    rnd = np.random.RandomState(100) # Using 100 as example seed
    #Noise Vars (DICT with numpy array)
    if importedNoiseVars is not None:
        #If noise vars provided - use provided
        #Make the dictionary
        varsToBeSet = {}
        #Open the json file
        with open(importedNoiseVars, 'r') as myfile:
            data=myfile.read()
        obj = json.loads(data)
        #Read each part of the json file into the dictionary
        for jsonVar in obj:
            #Need to convert back to dict, of form {tfvariable : numpy array}
            for tfVar in noise_vars:
                if tfVar.name == jsonVar:
                    varsToBeSet[tfVar] = np.array(obj[jsonVar])
    else:
        #If noise vars not provided - make random noise vars
        varsToBeSet = {var: rnd.randn(*var.shape.as_list()) for var in noise_vars}


    #Z variables (Multi dimensional array containing a single array of floats)
    if importedZVariables is not None:
        #If z variables provided - use provided
        #open json
        with open(importedZVariables, 'r') as myfile:
            data=myfile.read()
        obj = json.loads(data)
        #Read each part of the json file into the dictionary
        for var in obj:
            #Need to convert back to numpy array
            z = np.array(obj[var])
    else:
        #If z not provided - make z
        z = rnd.randn(1, *Gs.input_shape[1:])


    #Saving out noise vars to file
    myNoiseVarsDict = {}
    for var in varsToBeSet:
        myNoiseVarsDict[var.name] = varsToBeSet[var].tolist()
        write_json(f'{outdir}','myNoiseVarsDict.json',myNoiseVarsDict)

    #Saving out z variables to file
    myZVariableDict = {}
    myZVariableDict["z"] = z.tolist()
    write_json(f'{outdir}','z.json',myZVariableDict)

    #Save other logging details
    tflib.set_vars(varsToBeSet) # [height, width] setting random noise values, if theyre passed in then it will use them, save with random internal values and z value
    images = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
    #Generate the image
    PIL.Image.fromarray(images[0], 'RGB').save(f'{outdir}/MyImage.png')

#----------------------------------------------------------------------------
#Basically keep the code as it is
#And then with the generate function, you can have an optional dictionary that you pass in
#the names tf.Variable 'G_synthesis_1/noise0:0
#loop through the dict after noise_vars is generated and overwrite
#we would wanna be able to pass it in
#{var: rnd.randn(*var.shape.as_list()) for var in noise_vars} store in variable then modify it
#When we finish the function we want it to return the dictionary of resolution noise initialisation (AND SAVE IT OUT)
#if z isnt present then dont overwrite it
#also save z after

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:
  # Generate curated MetFaces images without truncation (Fig.10 left)
  python %(prog)s --outdir=out --trunc=1 --seeds=85,265,297,849 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl
  # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
  python %(prog)s --outdir=out --trunc=0.7 --seeds=600-605 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl
  # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
  python %(prog)s --outdir=out --trunc=1 --seeds=0-35 --class=1 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/cifar10.pkl
  # Render image from projected latent vector
  python %(prog)s --outdir=out --dlatents=out/dlatents.npz \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate images using pretrained network pickle.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    #add an argument that lets me specify a path of
    #Can load in json as dict as i need it
    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser.add_argument('--trunc', dest='truncation_psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser.add_argument('--class', dest='class_idx', type=int, help='Class label (default: unconditional)')
    parser.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')
    #My Jsons
    parser.add_argument('--noisevars', dest='importedNoiseVars', help='Imported json for reusing noise variables', metavar='DIR')
    parser.add_argument('--zvariables', dest='importedZVariables', help='Imported json for reusing z variables', metavar='DIR')
    args = parser.parse_args()
    generate_images(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
