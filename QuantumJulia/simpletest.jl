#Test time stepper
t,u ,v =timesteptest(0.1,0,2,"none",true)


@assert u == [-0.7425664611282105, 0.6697723873134257]
@assert v == [0.0, 0.0]

t,u ,v = timesteptest(0.1,1,2,"none",true)


@assert u ≈ [-0.6788190341608014, 0.0]
@assert v ≈ [0.0, -0.7330165520103384]

t,u ,v =timesteptest(0.1,2,2,"none",true)


@assert u ≈ [-0.6802294532422508, 0.0]
@assert v ≈ [0.0, -0.7327583389501704]

t,u ,v =timesteptest(0.1,3,2,"none",true)


@assert u ≈ [-0.6802921264320907, 3.437847696935607e-5]
@assert v ≈ [3.108295004389936e-5, -0.7328099457040584]

# Test splines
ctrl, g = bsplinetest(5,true)

@assert ctrl ≈ [0.0, -0.5, 0.0]
@assert g ≈ [ 0.23908612500000004,  0.71332775,  0.047586125000000035, 0.0, 0.0]


#test objfunc
N = 4
Nguard = 3
Ntot = N + Nguard
Ident = Matrix{Float64}(I, Ntot, Ntot)   
utarget = Ident[1:Ntot,1:N]
utarget[:,3] = Ident[:,4]
utarget[:,4] = Ident[:,3]
#utarget[:,1] = Ident[:,2]
#utarget[:,2] = Ident[:,1]
cfl = 0.05
T = 150.0
testadjoint = 0
maxpar =0.09
params = objfunc.parameters(N,Nguard,T,testadjoint,maxpar,cfl, utarget)
pcof = [1e-3, 2e-3, -2e-3]
order = 2

verbose= false
objv, grad  = objfunc.traceobjgrad(pcof, params, order, verbose, true)


@assert objv ≈ [0.6871073710795883]
@assert grad ≈ [0.043358677834784934,  0.05192466792999445,  -0.016074398822782755]