"""
Copied from: https://github.com/cctbx/cctbx_project/tree/master/simtbx
"""

from scitbx.array_family import flex
from simtbx.nanoBragg import testuple
from simtbx.nanoBragg import shapetype
from simtbx.nanoBragg import convention
from simtbx.nanoBragg import nanoBragg
import libtbx.load_env 
from cctbx import crystal
from cctbx import miller
assert miller
from sys import argv
import dxtbx

import time
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()








##############################################################################################################
print "loading ... "

def fcalc_from_pdb(resolution,algorithm=None,wavelength=0.977, pdb_lines = None):
    from iotbx import pdb
    pdb_inp = pdb.input(source_info=None,lines = pdb_lines)
    xray_structure = pdb_inp.xray_structure_simple()
    #
    # take a detour to insist on calculating anomalous contribution of every atom
    scatterers = xray_structure.scatterers()
    for sc in scatterers:
        from cctbx.eltbx import sasaki, henke
        #expected_sasaki = sasaki.table(sc.element_symbol()).at_angstrom(wavelength)
        expected_henke = henke.table(sc.element_symbol()).at_angstrom(wavelength)
        sc.fp = expected_henke.fp()
        sc.fdp = expected_henke.fdp()
    # how do we do bulk solvent?
    primitive_xray_structure = xray_structure.primitive_setting()
    P1_primitive_xray_structure = primitive_xray_structure.expand_to_p1()
    fcalc = P1_primitive_xray_structure.structure_factors(
      d_min=resolution, anomalous_flag=True, algorithm=algorithm).f_calc()
    return fcalc.amplitudes()
##############################################################################################################











##############################################################################################################
def run_sim2smv(fsave, pdb_lines, angle1, angle2, angle3, comm_rank, sfall=None):
    
    ## set detector size, pixel size, unit cell size
    SIM = nanoBragg(detpixels_slowfast=(1748, 1739), pixel_size_mm=0.11, Ncells_abc=(10,10,10), verbose=0)
    
    ## set detector distance
    SIM.distance_mm=138.695 #139
    
    
    # set wavelength and polarization
    SIM.seed = 1
    #SIM.randomize_orientation()
    SIM.missets_deg= (angle1, angle2, angle3)
    # SIM.oversample=1
    SIM.wavelength_A=0.977
    SIM.polarization=1
    
    
    # set default value
    # SIM.F000=1
    SIM.default_F=0
    

    ## calculate FHKL with pdb file
    # sfall = fcalc_from_pdb(resolution=1.3, algorithm="direct",wavelength=SIM.wavelength_A, pdb_lines = pdb_lines)
    
    # use crystal structure to initialize Fhkl array
    SIM.Fhkl=sfall
    
    # fastest option, least realistic
    SIM.xtal_shape=shapetype.Tophat
    
    # only really useful for long runs
    SIM.progress_meter=False
    
    # prints out value of one pixel only.  will not render full image!
    # SIM.show_params()
    
    # flux is always in photons/s
    SIM.flux = 5.0e11/50.0e-15  ###zhen ori 
    
    # assumes round beam
    SIM.beamsize_mm=1.0e-3
    SIM.exposure_s=50.0e-15
    
    
    SIM.beamcenter_convention=convention.ADXV
    SIM.beam_center_mm=(95.975, 96.855)  # 95.975 96.855
    
    
    # add Bragg peaks to simulated panel
    SIM.add_nanoBragg_spots()
    # amplify spot signal
    SIM.raw_pixels *= 117216 #400000   # raw pixel is amplitude, rather than intensity
    
    #print type(SIM.raw_pixels)
    
    # SIM.to_smv_format(fileout="only_bragg.img")
    
    
    
    # add background from water sin(theta/lambda) vs structure factor
    bg = flex.vec2_double([(0,2.57),(0.0365,2.58),(0.07,2.8),(0.12,5),(0.162,8),(0.2,6.75),(0.18,7.32),(0.216,6.75),(0.236,6.5),(0.28,4.5),(0.3,4.3),(0.345,4.36),(0.436,3.77),(0.5,3.17)])
    SIM.Fbg_vs_stol = bg
    SIM.amorphous_sample_thick_mm = 100.0e-3  ## 100um
    SIM.amorphous_density_gcm3 = 1
    SIM.amorphous_molecular_weight_Da = 18
    SIM.flux=5.0e11/50.0e-15
    SIM.beamsize_mm=1.0e-3
    SIM.exposure_s=50.0e-15
    SIM.add_background()
    # SIM.to_smv_format(fileout="bragg_and_water.img")
    
    
    # add background from air
    bg = flex.vec2_double([(0,14.1),(0.045,13.5),(0.174,8.35),(0.35,4.78),(0.5,4.22)])
    SIM.Fbg_vs_stol = bg
    SIM.amorphous_sample_thick_mm = 100 #35 # between beamstop and collimator
    SIM.amorphous_density_gcm3 = 1.0e-6 # 1.2e-3
    SIM.amorphous_sample_molecular_weight_Da = 28 # nitrogen = N2
    SIM.add_background()
    # SIM.to_smv_format(fileout="bragg_and_water_and_air.img")
    
    
    
    # set detector spread function.
    SIM.detector_psf_kernel_radius_pixels=1
    SIM.detector_psf_fwhm_mm=0.11
    SIM.detector_psf_type=shapetype.Gauss
    SIM.apply_psf()
    
    ## add noise
    SIM.add_noise()
    
    # image = SIM.raw_pixels.as_numpy_array()[::-1].T
    SIM.to_smv_format(fileout=fsave+"_temp_"+str(comm_rank).zfill(5)+".img")
    SIM.free_all()
    # return image
    return sfall
    

    
##############################################################################################################
def getpdbLines(filename):
    f = open(filename, "r")
    zzcontent = f.readlines()
    f.close()
    pdb_lines = "".join(zzcontent)
    pdb_lines = "CRYST1   65.000   70.000   75.000  90.00  90.00  90.00 P 21 21 21    4          \n" + pdb_lines
    #           "CRYST1   65.000   70.000   75.000  90.00  90.00  90.00 P 1           1          \n" + pdb_lines
    #           "CRYST1   65.000   70.000   75.000  90.00  90.00  90.00 P 21 21 21    4          \n" + pdb_lines
    #            CRYST1   37.444   47.982   82.984  90.00  90.00  90.00 C 2 2 21      8          
    return pdb_lines


##############################################################################################################
def write_h5py(fsave, idx, image):
    while True:
        try:
            f = h5py.File(fsave, "r+")
            f["eventNumber"][idx] = np.array([idx])
            f["data/data"][idx,:,:,:] = image
            f.close()
            return True
        except Exception as err:
            # print "ERROR :: ", err
            f = None
        time.sleep(0.1)
    
    
    
##############################################################################################################
def load_img(fname):
    f = open(fname,"rb")
    raw = f.read()
    h = raw[0:1024]
    d = raw[1024:]
    f.close()
    flat_image = np.frombuffer(d, dtype=np.uint16)
    simData = np.reshape(flat_image, ((1748, 1739)) ).astype(float)[::-1].T
    return simData


##############################################################################################################
fname = argv[1]  ## pdb
fsave = argv[2]  ## save to folder
random_euler = np.load("./random_euler_angles.npy")
Nsim = int(argv[3])
if len(argv) > 4:
    startidx = int(argv[4])
    rewrite = int(argv[5])
else:
    startidx = 0
    rewrite = 1
##############################################################################################################



##############################################################################################################
## setup
import h5py
import psana
experimentName = 'cxic0415'
runNumber = 100
detInfo = 'DscCsPad'
eventInd = 0

ds = psana.DataSource('exp='+str(experimentName)+':run='+str(runNumber)+':idx')
run = ds.runs().next()
times = run.times()
env = ds.env()
evt = run.event(times[0])
det = psana.Detector(str(detInfo), env)

evt = run.event(times[eventInd]) 
##
##############################################################################################################


if comm_rank == 0:
    if rewrite:
        print "rewrite the cxi file"
        f1=h5py.File(fsave, "w")
        dset=f1.create_dataset("data/data",shape=(Nsim,32,185,388),chunks=(1,32,185,388),dtype=np.float32)
        dset1=f1.create_dataset("eventNumber",shape=(Nsim,),dtype=np.int32) 
        # dset[...] = np.zeros((10000,32,185,388))
        # dset1[...] = np.arange(10000)
        f1.close()
    else:
        print "not rewrite the cxi file"

    
##############################################################################################################
print "starting ... "
## calculate sfall

pdb_lines = getpdbLines(fname)
sfall_out = fcalc_from_pdb(resolution=1.3, algorithm="direct",wavelength=0.977, pdb_lines = pdb_lines)

########################################################################################################






##############################################################################################################

for idx in range(Nsim):
    if idx%comm_size != comm_rank:
        continue
    if idx < startidx:
        continue
    
    angle1 = random_euler[idx, 0]
    angle2 = random_euler[idx, 1]
    angle3 = random_euler[idx, 2]
    
    print "%4d / %4d - is processing - %6d/%6d -> angle = (%f, %f, %f)"%(comm_rank, comm_size, idx, Nsim, angle1, angle2, angle3)
    
    run_sim2smv(fsave, pdb_lines, angle1, angle2, angle3, comm_rank, sfall=sfall_out)
    image = load_img(fsave+"_temp_"+str(comm_rank).zfill(5)+".img")
    # image[image<0] = 0
    print image.shape, np.amin(image), np.amax(image)
    
    stack = det.ndarray_from_image(par=evt, image=image, pix_scale_size_um=None, xy0_off_pix=None)
    write_h5py(fsave, idx, stack)