"""
Copied from: https://github.com/cctbx/cctbx_project/tree/master/simtbx
"""


from six.moves import range
from six.moves import StringIO
from scitbx.array_family import flex
from scitbx.matrix import sqr,col
from simtbx.nanoBragg import shapetype
from simtbx.nanoBragg import nanoBragg
import libtbx.load_env # possibly implicit
from cctbx import crystal
import math
import scitbx
from LS49.sim.util_fmodel import gen_fmodel
import os


import time
from omptbx import omp_get_num_procs

# %%% boilerplate specialize to packaged big data %%%
from LS49.sim import step4_pad
from LS49.spectra import generate_spectra
ls49_big_data = os.environ["LS49_BIG_DATA"] # get absolute path from environment
step4_pad.big_data = ls49_big_data
generate_spectra.big_data = ls49_big_data





def data(fpdb):
    return dict(
        pdb_lines = open(fpdb,"r").read()
    )


from LS49.sim.step4_pad import microcrystal


def write_safe(fname):
    # make sure file or compressed file is not already on disk
    return (not os.path.isfile(fname)) and (not os.path.isfile(fname+".gz"))


add_spots_algorithm = str(os.environ.get("ADD_SPOTS_ALGORITHM"))
def channel_pixels(wavelength_A,flux,N,UMAT_nm,Amatrix_rot,fmodel_generator,local_data,rank):
    fmodel_generator.reset_wavelength(wavelength_A)
    
    if rank==7: print("USING scatterer-specific energy-dependent scattering factors")
    sfall_channel = fmodel_generator.get_amplitudes()
    SIM = nanoBragg(detpixels_slowfast=(3000,3000),pixel_size_mm=0.11,Ncells_abc=(N,N,N),
        wavelength_A=wavelength_A,verbose=0)
    SIM.adc_offset_adu = 10 # Do not offset by 40
    SIM.mosaic_spread_deg = 0.05 # interpreted by UMAT_nm as a half-width stddev
    SIM.mosaic_domains = 25    # 77 seconds.    With 100 energy points, 7700 seconds (2 hours) per image
    SIM.distance_mm=141.7
    SIM.set_mosaic_blocks(UMAT_nm)

    # get same noise each time this test is run
    SIM.seed = 1
    SIM.oversample=1
    SIM.wavelength_A = wavelength_A
    SIM.polarization=1
    SIM.default_F=0
    SIM.Fhkl=sfall_channel
    SIM.Amatrix_RUB = Amatrix_rot
    SIM.xtal_shape=shapetype.Gauss # both crystal & RLP are Gaussian
    SIM.progress_meter=False
    # flux is always in photons/s
    SIM.flux=flux
    SIM.exposure_s=1.0 # so total fluence is e12
    # assumes round beam
    SIM.beamsize_mm=0.003 #cannot make this 3 microns; spots are too intense
    temp=SIM.Ncells_abc
    if rank==7: print("Ncells_abc=",SIM.Ncells_abc)
    SIM.Ncells_abc=temp

    from libtbx.development.timers import Profiler
    if rank==7: P = Profiler("nanoBragg C++ rank %d"%(rank))
    if add_spots_algorithm == "NKS":
        from boost.python import streambuf # will deposit printout into dummy StringIO as side effect
        SIM.add_nanoBragg_spots_nks(streambuf(StringIO()))
    elif add_spots_algorithm == "JH":
        SIM.add_nanoBragg_spots()
    elif add_spots_algorithm == "cuda":
        devices_per_node = int(os.environ["DEVICES_PER_NODE"])
        SIM.device_Id = rank%devices_per_node
        #if rank==7:
        #    os.system("nvidia-smi")
        SIM.add_nanoBragg_spots_cuda()
    else: raise Exception("!!! unknown spots algorithm")
    if rank==7: del P
    return SIM

from LS49.sim.debug_utils import channel_extractor
CHDBG_singleton = channel_extractor()

def run_sim2smv(prefix,crystal,spectra,rotation,rank,quick=False,save_bragg=False):
    local_data = data()
    smv_fileout = prefix + ".img"
    if not quick:
        if not write_safe(smv_fileout):
            print("File %s already exists, skipping in rank %d"%(smv_fileout,rank))
            return

    direct_algo_res_limit = 1.7

    wavlen, flux, wavelength_A = next(spectra) # list of lambdas, list of fluxes, average wavelength
    assert wavelength_A > 0


    GF = gen_fmodel(resolution=direct_algo_res_limit,pdb_text=local_data.get("pdb_lines"),algorithm="fft",wavelength=wavelength_A)
    GF.set_k_sol(0.435)
    GF.make_P1_primitive()
    sfall_main = GF.get_amplitudes()

    # use crystal structure to initialize Fhkl array
    sfall_main.show_summary(prefix = "Amplitudes used ")
    N = crystal.number_of_cells(sfall_main.unit_cell())

    #SIM = nanoBragg(detpixels_slowfast=(2000,2000),pixel_size_mm=0.11,Ncells_abc=(5,5,5),verbose=0)
    SIM = nanoBragg(detpixels_slowfast=(3000,3000),pixel_size_mm=0.11,Ncells_abc=(N,N,N),
        # workaround for problem with wavelength array, specify it separately in constructor.
        wavelength_A=wavelength_A,verbose=0)
    SIM.adc_offset_adu = 0 # Do not offset by 40
    SIM.adc_offset_adu = 10 # Do not offset by 40
    import sys
    if len(sys.argv)>2:
        SIM.seed = -int(sys.argv[2])
        print("GOTHERE seed=",SIM.seed)
    if len(sys.argv)>1:
        if sys.argv[1]=="random" : SIM.randomize_orientation()
    SIM.mosaic_spread_deg = 0.05 # interpreted by UMAT_nm as a half-width stddev
    SIM.mosaic_domains = 25    # 77 seconds.    
                                                     
    SIM.distance_mm=141.7

    UMAT_nm = flex.mat3_double()
    mersenne_twister = flex.mersenne_twister(seed=0)
    scitbx.random.set_random_seed(1234)
    rand_norm = scitbx.random.normal_distribution(mean=0, sigma=SIM.mosaic_spread_deg * math.pi/180.)
    g = scitbx.random.variate(rand_norm)
    mosaic_rotation = g(SIM.mosaic_domains)
    for m in mosaic_rotation:
        site = col(mersenne_twister.random_double_point_on_sphere())
        UMAT_nm.append( site.axis_and_angle_as_r3_rotation_matrix(m,deg=False) )
    SIM.set_mosaic_blocks(UMAT_nm)

    # get same noise each time this test is run
    SIM.seed = 1
    SIM.oversample=1
    SIM.wavelength_A = wavelength_A
    SIM.polarization=1
    # this will become F000, marking the beam center
    SIM.default_F=0
    #SIM.missets_deg= (10,20,30)
    
    SIM.Fhkl=sfall_main
    
    Amatrix_rot = (rotation * sqr(sfall_main.unit_cell().orthogonalization_matrix())).transpose()
    
    for i in Amatrix_rot: 
        print(i, end=' ')
    print()

    SIM.Amatrix_RUB = Amatrix_rot
    #workaround for failing init_cell, use custom written Amatrix setter
    
    Amat = sqr(SIM.Amatrix).transpose() # recovered Amatrix from SIM
    from cctbx import crystal_orientation
    Ori = crystal_orientation.crystal_orientation(Amat, crystal_orientation.basis_type.reciprocal)
    

    # fastest option, least realistic
    #SIM.xtal_shape=shapetype.Tophat # RLP = hard sphere
    #SIM.xtal_shape=shapetype.Square # gives fringes
    SIM.xtal_shape=shapetype.Gauss # both crystal & RLP are Gaussian
    #SIM.xtal_shape=shapetype.Round # Crystal is a hard sphere
    # only really useful for long runs
    SIM.progress_meter=False
    # prints out value of one pixel only.    will not render full image!
    #SIM.printout_pixel_fastslow=(500,500)
    #SIM.printout=True
    SIM.show_params()
    # flux is always in photons/s
    SIM.flux=1e12
    SIM.exposure_s=1.0 # so total fluence is e12
    # assumes round beam
    SIM.beamsize_mm=0.003 #cannot make this 3 microns; spots are too intense
    temp=SIM.Ncells_abc
    
    SIM.Ncells_abc=temp
    

    # simulated crystal is only 125 unit cells (25 nm wide)
    # amplify spot signal to simulate physical crystal of 4000x larger: 100 um (64e9 x the volume)
    print(crystal.domains_per_crystal)
    SIM.raw_pixels *= crystal.domains_per_crystal; # must calculate the correct scale!

    for x in range(len(flux)):
        from libtbx.development.timers import Profiler
        if rank==7: P = Profiler("nanoBragg Python and C++ rank %d"%(rank))

        if rank==7: print("+++++++++++++++++++++++++++++++++++++++ Wavelength",x)
        CH = channel_pixels(wavlen[x],flux[x],N,UMAT_nm,Amatrix_rot,GF,local_data,rank)
        SIM.raw_pixels += CH.raw_pixels * crystal.domains_per_crystal
        CHDBG_singleton.extract(channel_no=x, data=CH.raw_pixels)
        CH.free_all()

        if rank==7: del P
    

    # rough approximation to water: interpolation points for sin(theta/lambda) vs structure factor
    bg = flex.vec2_double([(0,2.57),(0.0365,2.58),(0.07,2.8),(0.12,5),(0.162,8),(0.2,6.75),(0.18,7.32),(0.216,6.75),(0.236,6.5),(0.28,4.5),(0.3,4.3),(0.345,4.36),(0.436,3.77),(0.5,3.17)])
    SIM.Fbg_vs_stol = bg
    SIM.amorphous_sample_thick_mm = 0.1
    SIM.amorphous_density_gcm3 = 1
    SIM.amorphous_molecular_weight_Da = 18
    SIM.flux=1e12
    SIM.beamsize_mm=0.003 # square (not user specified)
    SIM.exposure_s=1.0 # multiplies flux x exposure
    SIM.add_background()
    

    # rough approximation to air
    bg = flex.vec2_double([(0,14.1),(0.045,13.5),(0.174,8.35),(0.35,4.78),(0.5,4.22)])
    SIM.Fbg_vs_stol = bg
    #SIM.amorphous_sample_thick_mm = 35 # between beamstop and collimator
    SIM.amorphous_sample_thick_mm = 10 # between beamstop and collimator
    SIM.amorphous_density_gcm3 = 1.2e-3
    SIM.amorphous_sample_molecular_weight_Da = 28 # nitrogen = N2
    SIM.add_background()

    #apply beamstop mask here

    # set this to 0 or -1 to trigger automatic radius.    could be very slow with bright images
    # settings for CCD
    SIM.detector_psf_kernel_radius_pixels=5;
    #SIM.detector_psf_fwhm_mm=0.08;
    #SIM.detector_psf_type=shapetype.Fiber # rayonix=Fiber, CSPAD=None (or small Gaussian)
    SIM.detector_psf_type=shapetype.Unknown # for CSPAD
    SIM.detector_psf_fwhm_mm=0
    #SIM.apply_psf()

    SIM.add_noise() # converts phtons to ADU.

    extra = "PREFIX=%s;\nRANK=%d;\n"%(prefix,rank)
    SIM.to_smv_format_py(fileout=smv_fileout,intfile_scale=1,rotmat=True,extra=extra,gz=True)

    SIM.free_all()


def simulate_one(file_prefix, image, spectra, crystal, random_orientation, rank, sfall_cluster, quick=False):
    iterator = spectra.generate_recast_renormalized_image(image=image,energy=7120.,total_flux=1e12)

    rand_ori = sqr(random_orientation)
    print("random ori = ", rand_ori)
    run_sim2smv(prefix = file_prefix, crystal = crystal, spectra=iterator, rotation=rand_ori, quick=quick, rank=rank)


def sfall_prepare():
    sfall_cluster = {}

    return sfall_cluster


if __name__=="__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    import omptbx

    from sys import argv
    if len(argv)<=1:
        num_pdb = 150
        num_per_pdb = 10000
        N_total = num_pdb * num_per_pdb
    elif argv[1].lower()=="test":
        num_pdb = 2
        num_per_pdb = 10
        N_total = num_pdb * num_per_pdb
    else:
        raise Exception("!!! argument is wrong")

    workaround_nt = int(os.environ.get("OMP_NUM_THREADS",1))
    omptbx.omp_set_num_threads(workaround_nt)
    
    N_total = num_pdb * num_per_pdb
    N_stride = size # total number of worker tasks
    print("hello from rank %d of %d"%(rank,size),"with omp_threads=",omp_get_num_procs())
    
    import datetime
    start_elapse = time()

    if rank == 0:
        print("Rank 0 time", datetime.datetime.now())
        from LS49.spectra.generate_spectra import spectra_simulation

        SS = spectra_simulation()
        C = microcrystal(Deff_A = 4000, length_um = 4., beam_diameter_um = 1.0)   # assume smaller than 10 um crystals

        mt = flex.mersenne_twister(seed=0)
        random_orientations = []
        for iteration in range(N_total):
            random_orientations.append( mt.random_double_r3_rotation_matrix() )
            
        transmitted_info = dict(spectra = SS, crystal = C, random_orientations = random_orientations)
    else:
        transmitted_info = None

    transmitted_info = comm.bcast(transmitted_info, root = 0)
    comm.barrier()

    
    for idx_pdb in range(num_pdb):

        print("start processing pdb ", idx_pdb)

        sfall_cluster = None
        sfall_cluster = sfall_prepare()

        for idx_img in range(num_per_pdb):

            if idx_img % size == rank:

                idx_spectra = idx_pdb * num_per_pdb + idx_img
                # if rank==0: os.system("nvidia-smi")

                fsave = "lao_"+str(idx_pdb).zfill(3)+"_"+str(idx_img).zfill(6)+".img"
                
                print(rank, " ## idx_spectra = ", idx_spectra)
                print(rank, " ## idx_pdb = ", idx_pdb)
                print(rank, " ## idx_img = ", idx_img)
                print(rank, " ## fsave = ", fsave)

                if os.path.isfile(file_name):
                    continue

                simulate_one(fsave = fsave, image=idx, spectra=transmitted_info["spectra"],
                                crystal=transmitted_info["crystal"], random_orientation=transmitted_info["random_orientations"][idx], rank=rank, quick=False)
                

        sfall_cluster = None

    print("OK exiting rank",rank,"at",datetime.datetime.now())