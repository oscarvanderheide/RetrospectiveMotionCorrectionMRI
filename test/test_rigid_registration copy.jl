using RetrospectiveMotionCorrectionMRI, UtilitiesForMRI, JLD, LinearAlgebra

@load "./data/26-08-2022-volunteer-study/data/data_52763_motion1_priorT1_custom.jld"

opt = rigid_registration_options(; T=Float32, niter=10, η=1e-2, γ=0.92, verbose=true)
u_ = rigid_registration(ground_truth, prior, opt; spatial_geometry=X)