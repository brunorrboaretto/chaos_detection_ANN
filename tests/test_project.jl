#-- This code offer example tests of the chaos-noise distinction tool. It simulates dynamical systems, through the DynamicalSystems package in Julia, and tests 2 noisy signals.
#-- To run this, you need to have installed Julia (tested on version 1.5.4), and add the packages DynamicalSystems and Bridge. To do this: on the Julia terminal, type "]" then "add DynamicalSystems, Bridge". Also create dir time-series in the current dir The first run can take some time, but later runs are quick.
#-- Run through command: "julia test_project.jl"
#-- Outputs are the alpha returned by the ANN, the Omega, the symbolic entropy S and the flicker-noise entropy S_fn


using DynamicalSystems
using DelimitedFiles
using Bridge #for OU noise

DIR = "$(pwd())/.." #script installation directory. In this case, running from the tests dir

function getSubstringInString(char, string)
    string[findfirst(isequal(char), string)+1:end]
end

function getDistance(fileOut)
    script = "$(DIR)/auto_ANN_Omega/chaos_detection_ANN.py" #script path
    command = "python3 $(script) $(fileOut)"
    returnString = readlines(`sh -c $command`)[1]
    α, S, S_fn, Ω = parse.(Float64, getSubstringInString.('=', split(returnString)) ) #separates their values and parses to float
end

#Test time-series from a dynamical systems. Saves the file (simulating a normal input), and then runs the script on the file.
function testSystem(ds, systemName, texec)
    tr = trajectory(ds, texec)
    for i=1:size(tr,2)
        fileOut = "$(DIR)/tests/time-series/$(systemName)_var$(i)_exec_$(texec).dat"
        writedlm(fileOut, tr[:,i])
        α, S, S_fn, Ω = getDistance(fileOut)
        println("$(systemName) var $(i): α = $(α); Ω = $(Ω); S = $(S); S_fn = $(S_fn)")
    end
end

function testSystem_ts(ts, systemName, texec)
    fileOut = "$(DIR)/tests/time-series/$(systemName)_exec_$(texec).dat"
    writedlm(fileOut, ts)
    α, S, S_fn, Ω = getDistance(fileOut)
    println("$(systemName) var $(i): α = $(α); Ω = $(Ω); S = $(S); S_fn = $(S_fn)")
end


texec = 1000 #duration of time-series

# ---------------------------------------------------------------------- DYNAMICAL SYSTEMS  ---------------------------------------------------------------------- #
#--chua

ds= DynamicalSystemsBase.Systems.chua([0.7, 0., 0.]; a = 15.6, b = 25.58, m0 = -8/7, m1 = -5/7)
System = "Chua"
testSystem(ds, System, texec)

#--pomeau-mannevile
ds = DynamicalSystemsBase.Systems.pomeau_manneville(0.2, 2.5)
System = "Pomeau-Mannevile"
testSystem(ds, System, texec)

#--standard map
ds = DynamicalSystemsBase.Systems.standardmap(0.001*rand(2); k=0.971635)
System = "StandardMap"
testSystem(ds, System, texec)


#--double pendulum
ds = DynamicalSystemsBase.Systems.double_pendulum([π/2, 0, 0, rand()]; G=10.0, L1 = 1.0, L2 = 1.0, M1 = 1.0, M2 = 1.0)
System = "DoublePendulum"
testSystem(ds, System, texec)

#--henon
ds = DynamicalSystemsBase.Systems.henon(zeros(2); a = 1.4, b = 0.3)
System = "Henon"
testSystem(ds, System, texec)

#--towel map
ds = DynamicalSystemsBase.Systems.towel([0.085, -0.121, 0.075])
System = "Towel"
testSystem(ds, System, texec)



# --------------------------------------------------------------- NOISE --------------------------------------------------------------- #

# --ornstein-uhlenbeck

# Define a diffusion process
struct OrnsteinUhlenbeck  <: ContinuousTimeProcess{Float64}
    β::Float64 # drift parameter (also known as inverse relaxation time)
    σ::Float64 # diffusion parameter
end

# define drift and diffusion coefficient of OrnsteinUhlenbeck
Bridge.b(t, x, P::OrnsteinUhlenbeck) = -P.β*x
Bridge.σ(t, x, P::OrnsteinUhlenbeck) = P.σ

# simulate OrnsteinUhlenbeck using Euler scheme
# W = sample(0:0.01:10, Wiener())
W = sample(range(0, 10, length=texec), Wiener())
X = solve(EulerMaruyama(), 0.1, W, OrnsteinUhlenbeck(2.0, 1.0))
tr = [X[i][2] for i=2:length(X)]
systemName= "Ornstein-Uhlenbeck"; i = 1
# plot(tr)
testSystem_ts(tr, systemName, texec)


# --uniform noise
tr = rand(texec)
systemName= "UniformNoise";
testSystem_ts(tr, systemName, texec)
