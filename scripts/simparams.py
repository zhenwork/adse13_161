import os

## system params
#DEVICES_PER_NODE
quick = True                          ##################


## nanoBragg params
add_spots_algorithm = "JH"            ##################
fmodel_algorithm = "fft"
direct_algo_res_limit = 1.3
oversample = 1
default_F = 0


## crystal params
mosaic_spread_deg = 0.05
mosaic_domains = 25 
length_um = 10.
Deff_A = 4000
k_sol = 0.435


## background params
water_sample_thick_mm = 100.0e-3
water_density_gcm3 = 1
water_molecular_weight_Da = 18
air_sample_thick_mm = 100.0
air_density_gcm3 = 1.0e-6
air_molecular_weight_Da = 28  


## x-ray beam params
beam_diameter_um = 3.0 # 1.0
polarization = 1
wavelength_A = 0.977
energy_eV = 12690.348
exposure_s = 50.0e-15
beamsize_mm = 3.0e-3
flux = 5.0e11/50.0e-15


## device params
detector_size_nx = 1739
detector_size_ny = 1748
beam_center_x_mm = 95.975
beam_center_y_mm = 96.855
pixel_size_mm = 0.11
distance_mm = 138.695
detector_psf_kernel_radius_pixels = 1
detector_psf_fwhm_mm = 0.08


## user params
prefix = "lao"                                                                                   ##################
pdb_files = [ os.path.abspath( "./PDBs/lao_"+str(ii).zfill(3)+".pdb" ) for ii in range(2) ]      ##################
num_img = [5] * len(pdb_files)                                                                   ##################







