"""
Copied from: https://github.com/cctbx/cctbx_project/tree/master/simtbx
"""


from six.moves import range
from six.moves import StringIO
from scitbx.array_family import flex
from scitbx.matrix import sqr,col
from simtbx.nanoBragg import shapetype
from simtbx.nanoBragg import nanoBragg
from simtbx.nanoBragg import convention
import libtbx.load_env # possibly implicit
from cctbx import crystal
import math
import scitbx
from LS49.sim.util_fmodel import gen_fmodel
import os,sys


import time
import omptbx
from omptbx import omp_get_num_procs

# %%% boilerplate specialize to packaged big data %%%
from LS49.sim import step4_pad
from LS49.spectra import generate_spectra
ls49_big_data = os.environ["LS49_BIG_DATA"] # get absolute path from environment
step4_pad.big_data = ls49_big_data


import simparams as simparams
from LS49.sim.step4_pad import microcrystal
from cctbx import crystal_orientation



def data(fpdb):
    return dict(
        pdb_lines = open(fpdb,"r").read()
    )


def channel_pixels(simparams=None,single_wavelength_A=None,single_flux=None,N=None,UMAT_nm=None, \
                Amatrix_rot=None,sfall_channel=None,rank=None):
    
    # print("## inside channel wavlength/flux = ", single_wavelength_A, single_flux/simparams.flux)

    SIM = nanoBragg(detpixels_slowfast=(simparams.detector_size_ny,simparams.detector_size_nx),pixel_size_mm=simparams.pixel_size_mm,\
                Ncells_abc=(N,N,N),wavelength_A=single_wavelength_A,verbose=0)
    SIM.adc_offset_adu = 10      # Do not offset by 40
    SIM.seed = 0
    SIM.mosaic_spread_deg = simparams.mosaic_spread_deg # interpreted by UMAT_nm as a half-width stddev
    SIM.mosaic_domains = simparams.mosaic_domains      # 77 seconds.    With 100 energy points, 7700 seconds (2 hours) per image
    SIM.distance_mm=simparams.distance_mm
    SIM.set_mosaic_blocks(UMAT_nm)

    ######################
    SIM.beamcenter_convention=convention.ADXV
    SIM.beam_center_mm=(simparams.beam_center_x_mm, simparams.beam_center_y_mm)  # 95.975 96.855
    ######################

    # get same noise each time this test is run
    SIM.seed = 0
    SIM.oversample=simparams.oversample
    SIM.wavelength_A = single_wavelength_A
    SIM.polarization=simparams.polarization
    SIM.default_F=simparams.default_F
    SIM.Fhkl=sfall_channel
    SIM.Amatrix_RUB = Amatrix_rot
    SIM.xtal_shape=shapetype.Gauss # both crystal & RLP are Gaussian
    SIM.progress_meter=False
    # flux is always in photons/s
    SIM.flux=single_flux
    SIM.exposure_s=simparams.exposure_s # so total fluence is e12
    # assumes round beam
    SIM.beamsize_mm=simparams.beamsize_mm #cannot make this 3 microns; spots are too intense
    temp=SIM.Ncells_abc
    SIM.Ncells_abc=temp

    # SIM.show_params()

    add_spots_algorithm = simparams.add_spots_algorithm

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
    else: 
        raise Exception("!!! unknown spots algorithm")
    return SIM

from LS49.sim.debug_utils import channel_extractor
CHDBG_singleton = channel_extractor()

def run_sim2smv(img_prefix=None, simparams=None,pdb_lines=None,crystal=None,spectra=None,rotation=None,rank=None,fsave=None,sfall_cluster=None,quick=False):
    smv_fileout = fsave

    direct_algo_res_limit = simparams.direct_algo_res_limit

    wavlen, flux, real_wavelength_A = next(spectra) # list of lambdas, list of fluxes, average wavelength
    real_flux = flex.sum(flux)
    assert real_wavelength_A > 0
    # print(rank, " ## real_wavelength_A/real_flux = ", real_wavelength_A, real_flux*1.0/simparams.flux)

    if quick:
        wavlen = flex.double([real_wavelength_A])
        flux = flex.double([real_flux])

    # GF = gen_fmodel(resolution=simparams.direct_algo_res_limit,pdb_text=pdb_lines,algorithm=simparams.fmodel_algorithm,wavelength=real_wavelength_A)
    # GF.set_k_sol(simparams.k_sol) 
    # GF.make_P1_primitive()
    sfall_main = sfall_cluster["main"] #GF.get_amplitudes() 

    # use crystal structure to initialize Fhkl array
    # sfall_main.show_summary(prefix = "Amplitudes used ")
    N = crystal.number_of_cells(sfall_main.unit_cell())

    #print("## number of N = ", N)
    SIM = nanoBragg(detpixels_slowfast=(simparams.detector_size_ny,simparams.detector_size_nx),pixel_size_mm=simparams.pixel_size_mm,\
                Ncells_abc=(N,N,N),wavelength_A=real_wavelength_A,verbose=0)
        # workaround for problem with wavelength array, specify it separately in constructor.
        
    # SIM.adc_offset_adu = 0 # Do not offset by 40
    SIM.adc_offset_adu = 10 # Do not offset by 40
    
    
    SIM.seed = 0
    # SIM.randomize_orientation()
    SIM.mosaic_spread_deg = simparams.mosaic_spread_deg # interpreted by UMAT_nm as a half-width stddev
    SIM.mosaic_domains = simparams.mosaic_domains    # 77 seconds.                                                         
    SIM.distance_mm = simparams.distance_mm

    ## setup the mosaicity
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

    ######################
    SIM.beamcenter_convention=convention.ADXV
    SIM.beam_center_mm=(simparams.beam_center_x_mm, simparams.beam_center_y_mm)  # 95.975 96.855
    ######################

    # get same noise each time this test is run
    SIM.seed = 0
    SIM.oversample=simparams.oversample
    SIM.wavelength_A = real_wavelength_A
    SIM.polarization=simparams.polarization
    # this will become F000, marking the beam center
    SIM.default_F=simparams.default_F
    #SIM.missets_deg= (10,20,30)
    
    SIM.Fhkl=sfall_main
    
    Amatrix_rot = (rotation * sqr(sfall_main.unit_cell().orthogonalization_matrix())).transpose()

    SIM.Amatrix_RUB = Amatrix_rot
    #workaround for failing init_cell, use custom written Amatrix setter
    # print("## inside run_sim2smv, Amat_rot = ", Amatrix_rot)
    
    Amat = sqr(SIM.Amatrix).transpose() # recovered Amatrix from SIM
    
    Ori = crystal_orientation.crystal_orientation(Amat, crystal_orientation.basis_type.reciprocal)
    
    SIM.xtal_shape=shapetype.Gauss # both crystal & RLP are Gaussian

    SIM.progress_meter=False

    # SIM.show_params()
    # flux is always in photons/s
    SIM.flux=real_flux
    SIM.exposure_s=simparams.exposure_s 
    # assumes round beam
    SIM.beamsize_mm=simparams.beamsize_mm #cannot make this 3 microns; spots are too intense
    temp=SIM.Ncells_abc
    SIM.Ncells_abc=temp
    

    # print("## domains_per_crystal = ", crystal.domains_per_crystal)
    SIM.raw_pixels *= crystal.domains_per_crystal # must calculate the correct scale!

    # print("## Initial raw_pixels = ", flex.sum(SIM.raw_pixels))

    for x in range(len(flux)):
        # CH = channel_pixels(wavlen[x],flux[x],N,UMAT_nm,Amatrix_rot,sfall_cluster[x],rank)
        # print("## in loop wavlen/flux/real_wavelength_A = ", wavlen[x], flux[x]/real_flux, real_wavelength_A)
        CH = channel_pixels(simparams=simparams,single_wavelength_A=wavlen[x],single_flux=flux[x],N=N,UMAT_nm=UMAT_nm, \
                Amatrix_rot=Amatrix_rot,sfall_channel=sfall_cluster[x],rank=rank)
        SIM.raw_pixels += CH.raw_pixels * crystal.domains_per_crystal
        CHDBG_singleton.extract(channel_no=x, data=CH.raw_pixels)
        CH.free_all()
        # print("## sum raw_pixels after ", x, "is", flex.sum(SIM.raw_pixels))

        
    # rough approximation to water: interpolation points for sin(theta/lambda) vs structure factor
    bg = flex.vec2_double([(0,2.57),(0.0365,2.58),(0.07,2.8),(0.12,5),(0.162,8),(0.2,6.75),(0.18,7.32),(0.216,6.75),(0.236,6.5),(0.28,4.5),(0.3,4.3),(0.345,4.36),(0.436,3.77),(0.5,3.17)])
    SIM.Fbg_vs_stol = bg
    SIM.amorphous_sample_thick_mm = simparams.water_sample_thick_mm
    SIM.amorphous_density_gcm3 = simparams.water_density_gcm3
    SIM.amorphous_molecular_weight_Da = simparams.water_molecular_weight_Da
    SIM.flux=real_flux
    SIM.beamsize_mm=simparams.beamsize_mm   # square (not user specified)
    SIM.exposure_s=simparams.exposure_s     # multiplies flux x exposure
    SIM.add_background()
    

    # rough approximation to air
    bg = flex.vec2_double([(0,14.1),(0.045,13.5),(0.174,8.35),(0.35,4.78),(0.5,4.22)])
    SIM.Fbg_vs_stol = bg 
    SIM.amorphous_sample_thick_mm = simparams.air_sample_thick_mm # between beamstop and collimator
    SIM.amorphous_density_gcm3 = simparams.air_density_gcm3
    SIM.amorphous_sample_molecular_weight_Da = simparams.air_molecular_weight_Da  # nitrogen = N2
    SIM.add_background()

    #apply beamstop mask here

    # settings for CCD
    SIM.detector_psf_kernel_radius_pixels=simparams.detector_psf_kernel_radius_pixels
    SIM.detector_psf_type=shapetype.Unknown # for CSPAD
    SIM.detector_psf_fwhm_mm=simparams.detector_psf_fwhm_mm
    SIM.apply_psf()

    SIM.add_noise() 

    extra = "PREFIX=%s;\nRANK=%d;\n"%(img_prefix,rank)
    SIM.to_smv_format_py(fileout=smv_fileout,intfile_scale=1,rotmat=True,extra=extra,gz=True)
    SIM.free_all()



def sfall_prepare(simparams=None, fpdb=None, spectra=None):
    sfall_cluster = {}

    fmodel_generator = gen_fmodel(resolution=simparams.direct_algo_res_limit,\
                    pdb_text=data(fpdb).get("pdb_lines"), algorithm=simparams.fmodel_algorithm, wavelength=simparams.wavelength_A)
    fmodel_generator.set_k_sol(simparams.k_sol)
    fmodel_generator.make_P1_primitive()

    sfall_cluster["main"] = fmodel_generator.get_amplitudes().copy()

    if simparams.quick:
        sfall_cluster[0] = fmodel_generator.get_amplitudes().copy()
        return sfall_cluster

    iterator = spectra.generate_recast_renormalized_image(image=0, energy=simparams.energy_eV, total_flux=simparams.flux)
    wavlen, flux, real_wavelength_A = next(iterator)
    for x in range(len(flux)):
        print("## processing pdb with wavelength_A: ", wavlen[x], fpdb)
        fmodel_generator.reset_wavelength(wavlen[x])
        sfall_channel = fmodel_generator.get_amplitudes()
        sfall_cluster[ x ] = sfall_channel.copy()
    iterator = None
    return sfall_cluster


def jobAssign(size, num_pdb, num_img):
    
    # rank_in_pdb[rank][idx_pdb] = 0
    # size_in_pdb[idx_pdb] = 5
    # ranks_in_pdb[idx_pdb] = [0,3,5]
    # 
    
    rank_in_pdb = {}
    size_in_pdb = {}
    ranks_in_pdb = {}    

    for ipdb in range(num_pdb):
        size_in_pdb[ipdb] = 0
        ranks_in_pdb[ipdb] = []
    
    if ipdb >= size:
        for ipdb in range(num_pdb):
            r =  ipdb % size
            ranks_in_pdb[ipdb].append(r)
            size_in_pdb[ipdb] += 1
    else:
        for r in range(size):
            ipdb = r % num_pdb
            ranks_in_pdb[ipdb].append(r)
            size_in_pdb[ipdb] += 1
        
    for r in range(size):
        rank_in_pdb[r] = {}
    
    for ipdb in range(num_pdb):
        tmp_r = 0
        for r in ranks_in_pdb[ipdb]:
            rank_in_pdb[r][ipdb] = tmp_r
            tmp_r += 1
        
    return rank_in_pdb, size_in_pdb, ranks_in_pdb



if __name__=="__main__":

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    workaround_nt = int(os.environ.get("OMP_NUM_THREADS",1))
    omptbx.omp_set_num_threads(workaround_nt)
    print("## hello from rank %d of %d"%(rank,size),"with omp_threads=",omp_get_num_procs())
    

    ## assign jobs
    rank_in_pdb, size_in_pdb, ranks_in_pdb = jobAssign(size=size, num_pdb=simparams.num_pdbs, num_img=simparams.num_img[0])


    import datetime
    start_elapse = time.time()

    if rank == 0:
        print("Rank 0 time", datetime.datetime.now())
        from LS49.spectra.generate_spectra import spectra_simulation

        SS = spectra_simulation()
        C = microcrystal(Deff_A = simparams.Deff_A, length_um = simparams.length_um, beam_diameter_um = simparams.beam_diameter_um)   
        # assume smaller than 10 um crystals

        mt = flex.mersenne_twister(seed=0)
        random_orientations = []
        for iteration in range( sum(simparams.num_img) ):
            random_orientations.append( mt.random_double_r3_rotation_matrix() )
        
        # for ii in range(10): print("## TOP 10 orientations = ", random_orientations[ii])
        print("## total orientations = ", len(random_orientations))
        transmitted_info = dict(spectra = SS, crystal = C, random_orientations = random_orientations)

        for idx_pdb in range( len(simparams.pdb_files) ):
            save_folder = "./" + simparams.prefix + "_" + str(idx_pdb).zfill(3)
            if not os.path.isdir(save_folder):
                print("## creating folder: ", save_folder)
                os.makedirs(save_folder)
    else:
        transmitted_info = None

    transmitted_info = comm.bcast(transmitted_info, root = 0)
    comm.barrier()



    idx_img_all = -1
    for idx_pdb in range( len(simparams.pdb_files) ):

        fpdb = simparams.pdb_files[idx_pdb]
        
        sfall_cluster = None
        pdb_lines = None

        if rank in ranks_in_pdb[idx_pdb]:
            sfall_cluster = sfall_prepare(simparams=simparams, fpdb=fpdb, spectra = transmitted_info["spectra"])
            pdb_lines = data(fpdb).get("pdb_lines")
            save_folder = "./" + simparams.prefix + "_" + str(idx_pdb).zfill(3)

            print("## rank ", rank, " ## is precessing: ", fpdb, " ## PDB: ", pdb_lines[105:160])

        for idx_img_pdb in range(simparams.num_img[idx_pdb]):

            idx_img_all += 1 

            if rank not in ranks_in_pdb[idx_pdb]:
                continue

            if idx_img_pdb % size_in_pdb[idx_pdb] == rank_in_pdb[rank][idx_pdb]:

                fsave = save_folder + "/" + str(idx_img_pdb).zfill(6) + ".img"
                
                print("## rank ", rank, " is processing ", idx_pdb, " number: ", idx_img_pdb, " counted as: ", idx_img_all, " ## PDB: ", pdb_lines[105:160])

                if os.path.isfile(fsave) or os.path.isfile(fsave+".gz"):
                    print("@@ file exists: ", fsave)
                    continue

                spectra = transmitted_info["spectra"]
                iterator = None
                iterator = spectra.generate_recast_renormalized_image(image=idx_img_all%100005, energy=simparams.energy_eV, total_flux=simparams.flux)

                random_orientation=transmitted_info["random_orientations"][idx_img_all]
                rand_ori = sqr(random_orientation)
                # print("## rank ", rank, " ## random orientations = ", random_orientation)
                # print(rank, " ## random ori = ", rand_ori)
                run_sim2smv(img_prefix="PDB_"+str(idx_pdb).zfill(3),simparams=simparams,pdb_lines=pdb_lines,crystal=transmitted_info["crystal"],\
                        spectra=iterator,rotation=rand_ori,rank=rank,fsave=fsave,sfall_cluster=sfall_cluster,quick=simparams.quick)
                #run_sim2smv(prefix=simparams.prefix, crystal=transmitted_info["crystal"],spectra=iterator,rotation=rand_ori,\
                #            simparams=simparams,sfall_cluster=sfall_cluster,rank=rank,quick=simparams.quick)
                iterator = None

        sfall_cluster = None

    print("## OK exiting rank",rank,"at",datetime.datetime.now(), "elapsed", time.time()-start_elapse)