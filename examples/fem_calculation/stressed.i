[Mesh]


  [filem]
    type = FileMeshGenerator
	file = PrismsWithNamedSurfaces.inp
  []

  parallel_type = 'DISTRIBUTED'
  #parallel_type = 'REPLICATED'

[]

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
  volumetric_locking_correction = false
[]


#https://mooseframework.inl.gov/modules/tensor_mechanics/Dynamics.html
[AuxVariables]

  [./stress_xx]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./strain_xx]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_yy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./strain_yy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_zz]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./strain_zz]
    order = CONSTANT
    family = MONOMIAL
  [../]

[]

[ICs]
  [vel_z_IC]
    block = ball_TET4
    variable = vel_z
    value = -1.77  #  v = (2*9800/1e6*(160-0.1))^0.5   ball is dropping 160mm, but model starts at 0.1mm high
    type = ConstantIC
  []
[]


[Kernels]
  [./gravity]
    type = Gravity
    variable = disp_z
    value = -0.0098

  [../]
[]


[Modules/TensorMechanics/DynamicMaster]
  [all]
    add_variables = true

    displacements = 'disp_x disp_y disp_z'
    generate_output = 'stress_xx stress_yy stress_zz strain_xx strain_yy strain_zz vonmises_stress max_principal_strain'
    block = " ball_TET4 Volume2_TET4 Volume3_TET4"
    strain = FINITE
	verbose=True
	static_initialization=True
	incremental=True
  []
[]

[BCs]
  [./face_z]
    type = DirichletBC
    variable = disp_z
    boundary = BottomSurface
    value = 0
  [../]

  [./face_x]
    type = DirichletBC
    variable = disp_x
    boundary = BottomSurface
    value = 0
  [../]

  [./face_y]
    type = DirichletBC
    variable = disp_y
    boundary = BottomSurface
    value = 0
  [../]
[]


[Materials]
  [./acrylic]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 2.2e3
    poissons_ratio = 0.37
	block = ball_TET4
  [../]
   [./density_acrylic]
     type = GenericConstantMaterial
     block = ball_TET4
     prop_names = density
     prop_values = 1.190E-3
   [../]

  [./glass]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 70e3
    poissons_ratio = 0.24
    block = Volume3_TET4
  [../]
   [./density_glass]
     type = GenericConstantMaterial
     block = Volume3_TET4
     prop_names = density
     prop_values = 0.00262
   [../]
  [./PC]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 3e3
    poissons_ratio = 0.356
    block = Volume2_TET4
  [../]
  [./density_PC]
    type = GenericConstantMaterial
    block = Volume2_TET4
    prop_names = density
    prop_values = 0.0012006
  [../]

  [./stress]
    type = ComputeFiniteStrainElasticStress
  [../]

[]

[Contact]
  [ballcontact]
    secondary = TopSurface
    primary = ball_surface
    model = coulomb
    penalty = 1e+4
    friction_coefficient = 0.1

    formulation = penalty

    normalize_penalty = true
  []

  [inclusion]
    secondary = InnerSurfaceOuterPrism
    primary = OuterSurfaceInnerPrism
    model = glued
    penalty = 5e+4

    formulation = penalty
    normalize_penalty = true
  []
[]


[Preconditioning]
  [./smp]
    type = SMP
    full = true
  [../]
[]

[Executioner]
  type = Transient
  dt = 0.01
  solve_type = 'NEWTON'
  automatic_scaling = True

  dtmin = 1e-9
  dtmax = 0.01
  end_time = 0.5

  l_abs_tol = 1e-9
  nl_abs_tol = 5e-9

  l_max_its = 80
  nl_max_its = 10

  [TimeStepper]
    type = IterationAdaptiveDT
    optimal_iterations = 10
    dt = 0.01
	growth_factor = 1.5
  []
[]



[Outputs]
	exodus = true
  [csv]
    type = CSV
    execute_on = 'INITIAL TIMESTEP_END'
    file_base = stressed_csv
  []
[]


[Postprocessors]

  [./ball_velz]
    type = ElementAverageValue
    variable = vel_z
	block = ball_TET4
  [../]

  [./strain_princ_1]
	type = ElementExtremeValue
	variable = max_principal_strain
	value_type = max
  [../]

  [./strain_princ_block]
	type = ElementExtremeValue
	variable = max_principal_strain
	value_type = max
	block = Volume2_TET4
  [../]

  [./final_residual]
    type = Residual
    residual_type = final
  [../]
[]

[Debug]
  show_var_residual_norms=True
[]
